package kvcache

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

type turboquantEntry struct {
	key   []byte
	value []byte
}

type TurboQuantCache struct {
	meta             *Causal
	preset           turboquant.Preset
	requestedDType   ml.DType
	storageDType     ml.DType
	requestedBackend string
	data             map[int][]turboquantEntry
	shape            map[int]layerShape
}

type layerShape struct {
	keyDim     int
	valueDim   int
	numKVHeads int
}

func NewTurboQuantCache(base *Causal, preset turboquant.Preset, requestedBackend string) *TurboQuantCache {
	return &TurboQuantCache{
		meta:             base,
		preset:           preset,
		storageDType:     ml.DTypeF16,
		requestedBackend: strings.ToLower(strings.TrimSpace(requestedBackend)),
		data:             make(map[int][]turboquantEntry),
		shape:            make(map[int]layerShape),
	}
}

func WrapWithTurboQuant(cache Cache, preset turboquant.Preset, requestedBackend string) Cache {
	switch c := cache.(type) {
	case *TurboQuantCache:
		c.requestedBackend = strings.ToLower(strings.TrimSpace(requestedBackend))
		return c
	case *Causal:
		return NewTurboQuantCache(c, preset, requestedBackend)
	case *WrapperCache:
		for i := range c.caches {
			c.caches[i] = WrapWithTurboQuant(c.caches[i], preset, requestedBackend)
		}
		return c
	default:
		return cache
	}
}

func (c *TurboQuantCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	c.data = make(map[int][]turboquantEntry)
	c.shape = make(map[int]layerShape)
	c.requestedDType = dtype
	c.meta.Init(backend, c.storageDType, maxSequences, capacity, maxBatch)
}

func (c *TurboQuantCache) Close() {
	c.data = map[int][]turboquantEntry{}
	c.shape = map[int]layerShape{}
	c.meta.Close()
}

func (c *TurboQuantCache) SetLayer(layer int) {
	c.meta.SetLayer(layer)
}

func (c *TurboQuantCache) SetConfig(config ml.CacheConfig) {
	c.meta.SetConfig(config)
}

func (c *TurboQuantCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	return c.meta.StartForward(ctx, batch, reserve)
}

func (c *TurboQuantCache) Put(ctx ml.Context, key, value ml.Tensor) {
	kFloats := tensorToF32(ctx, key)
	vFloats := tensorToF32(ctx, value)

	layer := c.meta.curLayer
	c.ensureLayerStorage(layer)
	c.shape[layer] = layerShape{
		keyDim:     key.Dim(0),
		valueDim:   value.Dim(0),
		numKVHeads: key.Dim(1),
	}

	keyStride := key.Dim(0) * key.Dim(1)
	valueStride := value.Dim(0) * value.Dim(1)
	for i := 0; i < c.meta.curBatchSize; i++ {
		loc := c.meta.curLocs[i]

		keyBytes, err := c.encodeKeyVectorBytes(kFloats[i*keyStride:(i+1)*keyStride])
		if err != nil {
			panic(err)
		}

		valueBytes, err := c.encodeValueVectorBytes(vFloats[i*valueStride:(i+1)*valueStride])
		if err != nil {
			panic(err)
		}

		c.data[layer][loc] = turboquantEntry{
			key:   keyBytes,
			value: valueBytes,
		}
	}
}

func (c *TurboQuantCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	mask := c.meta.curMask
	layer := c.meta.curLayer
	layerEntries := c.data[layer]
	shape, ok := c.shape[layer]
	if !ok || len(layerEntries) == 0 || mask == nil {
		return nil, nil, mask
	}

	first := c.meta.curCellRange.min
	last := c.meta.curCellRange.max
	if first > last {
		return nil, nil, mask
	}

	cachedSize := mask.Dim(0)
	if keyTensor, valueTensor, ok := c.getFastPathTensors(ctx, layerEntries, shape, first, last, cachedSize); ok {
		return keyTensor, valueTensor, mask
	}

	keyData := make([]float32, shape.keyDim*shape.numKVHeads*cachedSize)
	valueData := make([]float32, shape.valueDim*shape.numKVHeads*cachedSize)

	for cell := first; cell <= last; cell++ {
		if cell < 0 || cell >= len(layerEntries) {
			continue
		}

		entry := layerEntries[cell]
		if len(entry.key) == 0 || len(entry.value) == 0 {
			continue
		}

		dst := cell - first
		decodedKey, _, err := turboquant.DecodeVector(entry.key)
		if err != nil {
			panic(err)
		}
		decodedValue, _, err := turboquant.DecodeVector(entry.value)
		if err != nil {
			panic(err)
		}

		if len(decodedKey) != shape.keyDim*shape.numKVHeads {
			panic(fmt.Sprintf("turboquant key decode shape mismatch: got %d want %d", len(decodedKey), shape.keyDim*shape.numKVHeads))
		}
		if len(decodedValue) != shape.valueDim*shape.numKVHeads {
			panic(fmt.Sprintf("turboquant value decode shape mismatch: got %d want %d", len(decodedValue), shape.valueDim*shape.numKVHeads))
		}

		copy(keyData[dst*shape.keyDim*shape.numKVHeads:(dst+1)*shape.keyDim*shape.numKVHeads], decodedKey)
		copy(valueData[dst*shape.valueDim*shape.numKVHeads:(dst+1)*shape.valueDim*shape.numKVHeads], decodedValue)
	}

	keyTensor := ctx.Input().FromFloats(keyData, shape.keyDim, shape.numKVHeads, cachedSize)
	if c.meta.config.PermutedV {
		return keyTensor, ctx.Input().FromFloats(
			permuteValueRows(valueData, shape.valueDim, shape.numKVHeads, cachedSize),
			cachedSize, shape.valueDim, shape.numKVHeads,
		), mask
	}

	return keyTensor, ctx.Input().FromFloats(valueData, shape.valueDim, shape.numKVHeads, cachedSize), mask
}

func (c *TurboQuantCache) getFastPathTensors(ctx ml.Context, layerEntries []turboquantEntry, shape layerShape, first, last, cachedSize int) (ml.Tensor, ml.Tensor, bool) {
	tqBackend, ok := c.meta.backend.(ml.TurboQuantBackend)
	if !ok {
		return nil, nil, false
	}

	support := tqBackend.TurboQuantSupport()
	switch c.requestedBackend {
	case "cuda":
		if !support.CUDA {
			return nil, nil, false
		}
	case "":
		if !support.CPU {
			return nil, nil, false
		}
	default:
		return nil, nil, false
	}

	rowBytes := 0
	keyBytes := make([]byte, 0)
	valueData := make([]float32, shape.valueDim*shape.numKVHeads*cachedSize)
	for cell := first; cell <= last; cell++ {
		if cell < 0 || cell >= len(layerEntries) {
			return nil, nil, false
		}

		entry := layerEntries[cell]
		if len(entry.key) == 0 || len(entry.value) == 0 {
			return nil, nil, false
		}

		if rowBytes == 0 {
			rowBytes = len(entry.key)
		} else if len(entry.key) != rowBytes {
			return nil, nil, false
		}

		keyBytes = append(keyBytes, entry.key...)

		dst := cell - first
		decodedValue, _, err := turboquant.DecodeVector(entry.value)
		if err != nil {
			return nil, nil, false
		}
		if len(decodedValue) != shape.valueDim*shape.numKVHeads {
			return nil, nil, false
		}
		copy(valueData[dst*shape.valueDim*shape.numKVHeads:(dst+1)*shape.valueDim*shape.numKVHeads], decodedValue)
	}

	if rowBytes == 0 {
		return nil, nil, false
	}

	keyTensor := ctx.Input().FromBytes(c.requestedDType, keyBytes, rowBytes, cachedSize)
	if c.meta.config.PermutedV {
		return keyTensor, ctx.Input().FromFloats(
			permuteValueRows(valueData, shape.valueDim, shape.numKVHeads, cachedSize),
			cachedSize, shape.valueDim, shape.numKVHeads,
		), true
	}

	return keyTensor, ctx.Input().FromFloats(valueData, shape.valueDim, shape.numKVHeads, cachedSize), true
}

func (c *TurboQuantCache) CopyPrefix(srcSeq, dstSeq int, len int32) {
	c.meta.CopyPrefix(srcSeq, dstSeq, len)
}

func (c *TurboQuantCache) CanResume(seq int, pos int32) bool {
	return c.meta.CanResume(seq, pos)
}

func (c *TurboQuantCache) Remove(seq int, beginIndex, endIndex int32) error {
	if endIndex != math.MaxInt32 && c.meta.shiftFn == nil {
		return ErrNotSupported
	}

	shifted := c.cellsRequiringShift(seq, endIndex)
	offset := beginIndex - endIndex

	if err := c.meta.Remove(seq, beginIndex, endIndex); err != nil {
		return err
	}

	if endIndex != math.MaxInt32 {
		if err := c.shiftPackedKeys(shifted, offset); err != nil {
			return err
		}
	}

	c.sweepReleasedCells()
	return nil
}

func (c *TurboQuantCache) ensureLayerStorage(layer int) {
	if _, ok := c.data[layer]; !ok {
		c.data[layer] = make([]turboquantEntry, len(c.meta.cells))
	}
}

func (c *TurboQuantCache) encodeKeyVectorBytes(values []float32) ([]byte, error) {
	encoded, err := turboquant.EncodeKeyVector(values, c.preset)
	if err != nil {
		return nil, err
	}

	return encoded.MarshalBinary()
}

func (c *TurboQuantCache) encodeValueVectorBytes(values []float32) ([]byte, error) {
	encoded, err := turboquant.EncodeValueVector(values, c.preset)
	if err != nil {
		return nil, err
	}

	return encoded.MarshalBinary()
}

func (c *TurboQuantCache) cellsRequiringShift(seq int, endIndex int32) []int {
	if endIndex == math.MaxInt32 {
		return nil
	}

	seqRange, ok := c.meta.cellRanges[seq]
	if !ok {
		return nil
	}

	var shifted []int
	for i := seqRange.min; i <= seqRange.max; i++ {
		cell := c.meta.cells[i]
		if !slices.Contains(cell.sequences, seq) || cell.pos < endIndex {
			continue
		}

		shifted = append(shifted, i)
	}

	return shifted
}

func (c *TurboQuantCache) shiftPackedKeys(cells []int, offset int32) error {
	if offset == 0 || len(cells) == 0 {
		return nil
	}

	shiftCtx := c.meta.backend.NewContext()
	defer shiftCtx.Close()

	shiftTensor := shiftCtx.Input().FromInts([]int32{offset}, 1)
	for layer, entries := range c.data {
		shape, ok := c.shape[layer]
		if !ok {
			continue
		}

		for _, cell := range cells {
			if cell < 0 || cell >= len(entries) || len(entries[cell].key) == 0 {
				continue
			}

			decodedKey, _, err := turboquant.DecodeVector(entries[cell].key)
			if err != nil {
				return err
			}

			keyTensor := shiftCtx.Input().FromFloats(decodedKey, shape.keyDim, shape.numKVHeads, 1)
			shiftedKey, err := c.meta.shiftFn(shiftCtx, layer, keyTensor, shiftTensor)
			if err != nil {
				return err
			}

			shiftedFloats := tensorToF32(shiftCtx, shiftedKey)
			keyBytes, err := c.encodeKeyVectorBytes(shiftedFloats)
			if err != nil {
				return err
			}

			entries[cell].key = keyBytes
		}

		c.data[layer] = entries
	}

	return nil
}

func (c *TurboQuantCache) sweepReleasedCells() {
	for layer, entries := range c.data {
		for i := range entries {
			if len(c.meta.cells[i].sequences) == 0 {
				entries[i] = turboquantEntry{}
			}
		}
		c.data[layer] = entries
	}
}

func permuteValueRows(values []float32, valueDim, numKVHeads, cachedSize int) []float32 {
	permuted := make([]float32, len(values))
	for cell := 0; cell < cachedSize; cell++ {
		for kv := 0; kv < numKVHeads; kv++ {
			for head := 0; head < valueDim; head++ {
				src := cell*valueDim*numKVHeads + kv*valueDim + head
				dst := cell + cachedSize*head + cachedSize*valueDim*kv
				permuted[dst] = values[src]
			}
		}
	}

	return permuted
}

func tensorToF32(ctx ml.Context, t ml.Tensor) []float32 {
	f32 := t
	if t.DType() != ml.DTypeF32 {
		f32 = ctx.Input().Empty(ml.DTypeF32, t.Shape()...)
		f32 = t.Copy(ctx, f32)
	}
	ctx.Forward(f32).Compute(f32)
	return append([]float32(nil), f32.Floats()...)
}

func presetFromDType(dtype ml.DType) (turboquant.Preset, bool) {
	switch dtype {
	case ml.DTypeTQ25:
		return turboquant.PresetTQ25, true
	case ml.DTypeTQ35:
		return turboquant.PresetTQ35, true
	default:
		return turboquant.Preset{}, false
	}
}
