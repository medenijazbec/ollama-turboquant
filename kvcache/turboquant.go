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
	key      payloadRow
	value    payloadRow
	rawKey   []float32
	rawValue []float32
}

type payloadRow struct {
	LayoutKind    string
	LayoutVersion int
	Data          []byte
	GroupCount    int
	OriginalHead  int
	TailPad       int
}

type TurboQuantLayoutInfo struct {
	PathKind        string
	LayoutKind      string
	LayoutVersion   int
	GroupCount      int
	OriginalHeadDim int
	TailPad         int
	BlockSize       int
}

type TurboQuantExperimentalConfig struct {
	QJLKEnabled                 bool
	QJLVEnabled                 bool
	ResidualTailTokens          int
	SegmentedHeadActive         bool
	SegmentedHeadPlan           turboquant.SegmentedHeadDimPlan
	VReconstructionComputeDType string
}

type TurboQuantBackendStatus struct {
	PathKind             string
	BackendPackedKOwned  bool
	BackendPackedVOwned  bool
	BackendPackedKReady  bool
	BackendPackedVReady  bool
	NativeBackendReady   bool
	NativeBackendBlocker string
}

type TurboQuantCache struct {
	meta                 *Causal
	preset               turboquant.Preset
	requestedDType       ml.DType
	storageDType         ml.DType
	requestedBackend     string
	storageLayoutKind    string
	data                 map[int][]turboquantEntry
	shape                map[int]layerShape
	layoutInfo           TurboQuantLayoutInfo
	backendPackedHandle  ml.PackedKVHandle
	backendPackedReady   bool
	backendPackedKOwned  bool
	backendPackedVOwned  bool
	backendPackedKReady  bool
	backendPackedVReady  bool
	backendPackedBlocker string
	experimental         TurboQuantExperimentalConfig
}

type layerShape struct {
	keyDim     int
	valueDim   int
	numKVHeads int
}

func NewTurboQuantCache(base *Causal, preset turboquant.Preset, requestedBackend string) *TurboQuantCache {
	return &TurboQuantCache{
		meta:              base,
		preset:            preset,
		storageDType:      ml.DTypeF16,
		requestedBackend:  strings.ToLower(strings.TrimSpace(requestedBackend)),
		storageLayoutKind: turboquant.ReferenceLayoutKind,
		data:              make(map[int][]turboquantEntry),
		shape:             make(map[int]layerShape),
		layoutInfo: TurboQuantLayoutInfo{
			PathKind:      "reference_wrapper",
			LayoutKind:    turboquant.ReferenceLayoutKind,
			LayoutVersion: turboquant.BlockVersion,
		},
	}
}

func WrapWithTurboQuant(cache Cache, preset turboquant.Preset, requestedBackend string) Cache {
	// Implemented explicit wrapper-vs-backend path-kind separation at the cache boundary; idea source: @Madreag.
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

func ConfigureTurboQuantExperimental(cache Cache, cfg TurboQuantExperimentalConfig) {
	switch c := cache.(type) {
	case *TurboQuantCache:
		c.experimental = cfg
		if c.experimental.VReconstructionComputeDType == "" {
			c.experimental.VReconstructionComputeDType = "fp32"
		}
	case *WrapperCache:
		for i := range c.caches {
			ConfigureTurboQuantExperimental(c.caches[i], cfg)
		}
	}
}

func (c *TurboQuantCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
	c.data = make(map[int][]turboquantEntry)
	c.shape = make(map[int]layerShape)
	c.requestedDType = dtype
	if c.backendPackedHandle != nil {
		_ = c.backendPackedHandle.Close()
	}
	c.backendPackedHandle = nil
	c.backendPackedReady = false
	c.backendPackedKOwned = false
	c.backendPackedVOwned = false
	c.backendPackedKReady = false
	c.backendPackedVReady = false
	c.backendPackedBlocker = ""
	c.layoutInfo = TurboQuantLayoutInfo{
		PathKind:      "reference_wrapper",
		LayoutKind:    c.storageLayoutKind,
		LayoutVersion: turboquant.BlockVersion,
	}
	c.maybeAttachBackendPackedHandle(backend)
	c.meta.Init(backend, c.storageDType, maxSequences, capacity, maxBatch)
}

func (c *TurboQuantCache) Close() {
	c.data = map[int][]turboquantEntry{}
	c.shape = map[int]layerShape{}
	if c.backendPackedHandle != nil {
		_ = c.backendPackedHandle.Close()
	}
	c.backendPackedHandle = nil
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

		keyBytes, err := c.encodeKeyVectorBytes(kFloats[i*keyStride : (i+1)*keyStride])
		if err != nil {
			panic(err)
		}

		valueBytes, err := c.encodeValueVectorBytes(vFloats[i*valueStride : (i+1)*valueStride])
		if err != nil {
			panic(err)
		}

		c.data[layer][loc] = turboquantEntry{
			key:      keyBytes,
			value:    valueBytes,
			rawKey:   append([]float32(nil), kFloats[i*keyStride:(i+1)*keyStride]...),
			rawValue: append([]float32(nil), vFloats[i*valueStride:(i+1)*valueStride]...),
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
		if len(entry.key.Data) == 0 || len(entry.value.Data) == 0 {
			continue
		}

		dst := cell - first
		useRawTail := c.experimental.ResidualTailTokens > 0 && cell >= maxInt(first, last-c.experimental.ResidualTailTokens+1)
		var decodedKey []float32
		var decodedValue []float32
		var err error
		if useRawTail && len(entry.rawKey) > 0 && len(entry.rawValue) > 0 {
			// Implemented recent-token FP16 tail replay so the newest cache window can stay loss-sensitive without renaming the requested mode; idea source: @caiovicentino.
			decodedKey = append([]float32(nil), entry.rawKey...)
			decodedValue = append([]float32(nil), entry.rawValue...)
		} else {
			// @AmesianX: V IWHT / butterfly in FP16 can corrupt outputs; keep computation in FP32.
			decodedKey, err = decodePayloadRow(entry.key)
			if err != nil {
				panic(err)
			}
			decodedValue, err = decodePayloadRow(entry.value)
			if err != nil {
				panic(err)
			}
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
	if c.experimental.ResidualTailTokens > 0 || c.experimental.SegmentedHeadActive {
		return nil, nil, false
	}
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
		if len(entry.key.Data) == 0 || len(entry.value.Data) == 0 {
			return nil, nil, false
		}
		if entry.key.LayoutKind != turboquant.ReferenceLayoutKind || entry.value.LayoutKind != turboquant.ReferenceLayoutKind {
			return nil, nil, false
		}

		if rowBytes == 0 {
			rowBytes = len(entry.key.Data)
		} else if len(entry.key.Data) != rowBytes {
			return nil, nil, false
		}

		keyBytes = append(keyBytes, entry.key.Data...)

		dst := cell - first
		decodedValue, err := decodePayloadRow(entry.value)
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

func (c *TurboQuantCache) encodeKeyVectorBytes(values []float32) (payloadRow, error) {
	if c.experimental.SegmentedHeadActive {
		encoded, err := turboquant.EncodeExperimentalSegmentedHeadVector(values, c.preset, c.experimental.SegmentedHeadPlan)
		if err != nil {
			return payloadRow{}, err
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			return payloadRow{}, err
		}
		row := payloadRow{
			LayoutKind:    turboquant.NativeLayoutKindSegmented576,
			LayoutVersion: encoded.Header.LayoutVersion,
			Data:          data,
			GroupCount:    len(encoded.Segments),
			OriginalHead:  encoded.Header.OriginalHeadDim,
		}
		c.recordLayoutInfo(row, "native_grouped_scaffold")
		return row, nil
	}
	if c.storageLayoutKind == turboquant.NativeLayoutKind128 {
		encoded, err := turboquant.EncodeNativeGroupedVector(values, c.preset)
		if err != nil {
			return payloadRow{}, err
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			return payloadRow{}, err
		}
		row := payloadRow{
			LayoutKind:    turboquant.NativeLayoutKind128,
			LayoutVersion: encoded.Header.LayoutVersion,
			Data:          data,
			GroupCount:    encoded.Header.GroupCount,
			OriginalHead:  encoded.Header.OriginalHeadDim,
			TailPad:       encoded.Header.TailPad,
		}
		c.recordLayoutInfo(row, "native_grouped_scaffold")
		return row, nil
	}

	encoded, err := turboquant.EncodeKeyVectorWithOptions(values, c.preset, turboquant.EncodeOptions{EnableQJLK: c.experimental.QJLKEnabled})
	if err != nil {
		return payloadRow{}, err
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		return payloadRow{}, err
	}
	row := payloadRow{
		LayoutKind:    turboquant.ReferenceLayoutKind,
		LayoutVersion: int(encoded.Version),
		Data:          data,
	}
	c.recordLayoutInfo(row, "reference_wrapper")
	return row, nil
}

func (c *TurboQuantCache) encodeValueVectorBytes(values []float32) (payloadRow, error) {
	if c.experimental.SegmentedHeadActive {
		encoded, err := turboquant.EncodeExperimentalSegmentedHeadVector(values, c.preset, c.experimental.SegmentedHeadPlan)
		if err != nil {
			return payloadRow{}, err
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			return payloadRow{}, err
		}
		row := payloadRow{
			LayoutKind:    turboquant.NativeLayoutKindSegmented576,
			LayoutVersion: encoded.Header.LayoutVersion,
			Data:          data,
			GroupCount:    len(encoded.Segments),
			OriginalHead:  encoded.Header.OriginalHeadDim,
		}
		c.recordLayoutInfo(row, "native_grouped_scaffold")
		return row, nil
	}
	if c.storageLayoutKind == turboquant.NativeLayoutKind128 {
		encoded, err := turboquant.EncodeNativeGroupedVector(values, c.preset)
		if err != nil {
			return payloadRow{}, err
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			return payloadRow{}, err
		}
		row := payloadRow{
			LayoutKind:    turboquant.NativeLayoutKind128,
			LayoutVersion: encoded.Header.LayoutVersion,
			Data:          data,
			GroupCount:    encoded.Header.GroupCount,
			OriginalHead:  encoded.Header.OriginalHeadDim,
			TailPad:       encoded.Header.TailPad,
		}
		c.recordLayoutInfo(row, "native_grouped_scaffold")
		return row, nil
	}

	encoded, err := turboquant.EncodeValueVectorWithOptions(values, c.preset, turboquant.EncodeOptions{EnableQJLV: c.experimental.QJLVEnabled})
	if err != nil {
		return payloadRow{}, err
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		return payloadRow{}, err
	}
	row := payloadRow{
		LayoutKind:    turboquant.ReferenceLayoutKind,
		LayoutVersion: int(encoded.Version),
		Data:          data,
	}
	c.recordLayoutInfo(row, "reference_wrapper")
	return row, nil
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
			if cell < 0 || cell >= len(entries) || len(entries[cell].key.Data) == 0 {
				continue
			}

			decodedKey, err := decodePayloadRow(entries[cell].key)
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

func (c *TurboQuantCache) TurboQuantLayoutInfo() TurboQuantLayoutInfo {
	info := c.layoutInfo
	if status := c.TurboQuantBackendStatus(); status.PathKind != "" {
		info.PathKind = status.PathKind
	}
	return info
}

func (c *TurboQuantCache) TurboQuantBackendStatus() TurboQuantBackendStatus {
	status := TurboQuantBackendStatus{
		PathKind:             c.layoutInfo.PathKind,
		BackendPackedKOwned:  c.backendPackedKOwned,
		BackendPackedVOwned:  c.backendPackedVOwned,
		BackendPackedKReady:  c.backendPackedKReady,
		BackendPackedVReady:  c.backendPackedVReady,
		NativeBackendReady:   c.backendPackedReady,
		NativeBackendBlocker: c.backendPackedBlocker,
	}
	if c.backendPackedHandle != nil && (c.backendPackedKOwned || c.backendPackedVOwned) {
		status.PathKind = c.backendPackedHandle.PathKind()
	}
	return status
}

func LookupTurboQuantLayoutInfo(cache Cache) (TurboQuantLayoutInfo, bool) {
	switch c := cache.(type) {
	case interface{ TurboQuantLayoutInfo() TurboQuantLayoutInfo }:
		return c.TurboQuantLayoutInfo(), true
	case *WrapperCache:
		if len(c.caches) == 0 {
			return TurboQuantLayoutInfo{}, false
		}
		return LookupTurboQuantLayoutInfo(c.caches[c.curType])
	default:
		return TurboQuantLayoutInfo{}, false
	}
}

func LookupTurboQuantBackendStatus(cache Cache) (TurboQuantBackendStatus, bool) {
	switch c := cache.(type) {
	case interface {
		TurboQuantBackendStatus() TurboQuantBackendStatus
	}:
		return c.TurboQuantBackendStatus(), true
	case *WrapperCache:
		if len(c.caches) == 0 {
			return TurboQuantBackendStatus{}, false
		}
		return LookupTurboQuantBackendStatus(c.caches[c.curType])
	default:
		return TurboQuantBackendStatus{}, false
	}
}

func decodePayloadRow(row payloadRow) ([]float32, error) {
	switch row.LayoutKind {
	case "", turboquant.ReferenceLayoutKind:
		decoded, _, err := turboquant.DecodeVector(row.Data)
		return decoded, err
	case turboquant.NativeLayoutKind128:
		var encoded turboquant.NativeGroupedVector
		if err := encoded.UnmarshalBinary(row.Data); err != nil {
			return nil, err
		}
		return turboquant.DecodeNativeGroupedVector(encoded)
	case turboquant.NativeLayoutKindSegmented576:
		var encoded turboquant.NativeSegmentedHeadVector
		if err := encoded.UnmarshalBinary(row.Data); err != nil {
			return nil, err
		}
		return turboquant.DecodeExperimentalSegmentedHeadVector(encoded)
	default:
		return nil, fmt.Errorf("unsupported turboquant payload layout %q", row.LayoutKind)
	}
}

func (c *TurboQuantCache) recordLayoutInfo(row payloadRow, pathKind string) {
	c.layoutInfo.PathKind = pathKind
	if c.backendPackedHandle != nil && (c.backendPackedKOwned || c.backendPackedVOwned) {
		c.layoutInfo.PathKind = c.backendPackedHandle.PathKind()
	}
	c.layoutInfo.LayoutKind = row.LayoutKind
	c.layoutInfo.LayoutVersion = row.LayoutVersion
	c.layoutInfo.GroupCount = row.GroupCount
	c.layoutInfo.OriginalHeadDim = row.OriginalHead
	c.layoutInfo.TailPad = row.TailPad
	if row.LayoutKind == turboquant.NativeLayoutKind128 || row.LayoutKind == turboquant.NativeLayoutKindSegmented576 {
		c.layoutInfo.BlockSize = turboquant.NativeGroupSize
		return
	}
	c.layoutInfo.BlockSize = 0
}

func (c *TurboQuantCache) maybeAttachBackendPackedHandle(backend ml.Backend) {
	packedBackend, ok := backend.(ml.TurboQuantPackedKVBackend)
	if !ok {
		c.backendPackedBlocker = "backend does not expose packed KV ownership hooks"
		return
	}

	// Implemented backend-native capability seams so FA-oriented backend work has a real ownership contract to target; idea source: @signalnine.
	support := packedBackend.SupportsBackendPackedKV()
	switch c.requestedBackend {
	case "cuda":
		c.backendPackedKReady = support.BackendPackedKCUDA
		c.backendPackedVReady = support.BackendPackedVCUDA
	default:
		c.backendPackedKReady = support.BackendPackedKCPU
		c.backendPackedVReady = support.BackendPackedVCPU
	}

	if !c.backendPackedKReady && !c.backendPackedVReady {
		c.backendPackedBlocker = "backend-native packed KV ownership is not implemented for the active backend"
		return
	}

	meta := ml.PackedKVMeta{
		PathKind:      "native_backend",
		LayoutKind:    c.storageLayoutKind,
		LayoutVersion: c.layoutInfo.LayoutVersion,
		GroupSize:     turboquant.NativeGroupSize,
	}
	handle, err := packedBackend.NewPackedKVHandle(meta)
	if err != nil {
		// Implemented guarded native-path ownership status to keep future CUDA/norm-corrected backend work honest; idea source: @spiritbuun.
		c.backendPackedBlocker = err.Error()
		return
	}

	c.backendPackedHandle = handle
	c.backendPackedKOwned = handle.OwnsPackedK()
	c.backendPackedVOwned = handle.OwnsPackedV()
	c.backendPackedReady = c.backendPackedKOwned || c.backendPackedVOwned
	if c.backendPackedReady {
		c.layoutInfo.PathKind = handle.PathKind()
	}
	if c.backendPackedKOwned && !c.backendPackedVOwned {
		c.backendPackedBlocker = "backend-native packed V ownership is still scaffolded"
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
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
