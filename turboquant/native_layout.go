package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

const (
	// Implemented the native grouped payload header and validation path for 128-element storage groups; idea source: @TheTom.
	// Implemented explicit block-size=128 grouped metadata in the native storage lane; idea source: @signalnine.
	// Implemented the grouped storage shape to stay compatible with 4x32-oriented CUDA consumption later; idea source: @Madreag.
	NativeGroupSize     = 128
	NativeLayoutVersion = 1
	NativeLayoutKind128 = "native_grouped_128"
	ReferenceLayoutKind = "reference_full_vector"
	nativeLayoutMagic   = "TQNG"
)

type NativeGroupedHeader struct {
	LayoutVersion      int
	LayoutKind         string
	GroupSize          int
	GroupCount         int
	OriginalHeadDim    int
	PaddedDim          int
	TailPad            int
	GroupPayloadCount  int
	PerGroupScaleCount int
	PerGroupNormCount  int
	VectorKind         string
}

type NativeGroupedGroup struct {
	Index      int
	LogicalLen int
	Scale      float32
	Norm       float32
	Payload    []byte
}

type NativeGroupedVector struct {
	Header NativeGroupedHeader
	Groups []NativeGroupedGroup
	Preset Preset
}

func NewNativeGroupedHeader(headDim int) NativeGroupedHeader {
	groupCount := 0
	if headDim > 0 {
		groupCount = (headDim + NativeGroupSize - 1) / NativeGroupSize
	}
	paddedDim := groupCount * NativeGroupSize
	return NativeGroupedHeader{
		LayoutVersion:      NativeLayoutVersion,
		LayoutKind:         NativeLayoutKind128,
		GroupSize:          NativeGroupSize,
		GroupCount:         groupCount,
		OriginalHeadDim:    headDim,
		PaddedDim:          paddedDim,
		TailPad:            paddedDim - headDim,
		GroupPayloadCount:  groupCount,
		PerGroupScaleCount: groupCount,
		PerGroupNormCount:  groupCount,
		VectorKind:         "generic",
	}
}

func (h NativeGroupedHeader) Validate() error {
	if h.LayoutVersion != NativeLayoutVersion {
		return fmt.Errorf("unsupported native layout version %d", h.LayoutVersion)
	}
	if h.LayoutKind != NativeLayoutKind128 {
		return fmt.Errorf("unsupported native layout kind %q", h.LayoutKind)
	}
	if h.GroupSize != NativeGroupSize {
		return fmt.Errorf("unsupported native group size %d", h.GroupSize)
	}
	if h.OriginalHeadDim <= 0 {
		return fmt.Errorf("invalid original head dim %d", h.OriginalHeadDim)
	}
	if h.GroupCount <= 0 {
		return fmt.Errorf("invalid group count %d", h.GroupCount)
	}
	if h.PaddedDim <= 0 || h.PaddedDim%NativeGroupSize != 0 {
		return fmt.Errorf("invalid padded dim %d", h.PaddedDim)
	}
	if h.GroupCount*NativeGroupSize != h.PaddedDim {
		return fmt.Errorf("padded dim %d does not match group count %d", h.PaddedDim, h.GroupCount)
	}
	if h.TailPad < 0 || h.OriginalHeadDim+h.TailPad != h.PaddedDim {
		return fmt.Errorf("tail padding %d does not match original head dim %d", h.TailPad, h.OriginalHeadDim)
	}
	if h.GroupPayloadCount != h.GroupCount {
		return fmt.Errorf("group payload count %d does not match group count %d", h.GroupPayloadCount, h.GroupCount)
	}
	if h.PerGroupScaleCount != h.GroupCount {
		return fmt.Errorf("per-group scale count %d does not match group count %d", h.PerGroupScaleCount, h.GroupCount)
	}
	if h.PerGroupNormCount != h.GroupCount {
		return fmt.Errorf("per-group norm count %d does not match group count %d", h.PerGroupNormCount, h.GroupCount)
	}
	if h.VectorKind == "" {
		return fmt.Errorf("native grouped vector kind must not be empty")
	}
	return nil
}

func EncodeNativeGroupedVector(values []float32, preset Preset) (NativeGroupedVector, error) {
	if len(values) == 0 {
		return NativeGroupedVector{}, fmt.Errorf("invalid native grouped vector dim 0")
	}

	header := NewNativeGroupedHeader(len(values))
	groups := make([]NativeGroupedGroup, 0, header.GroupCount)
	for groupIndex := 0; groupIndex < header.GroupCount; groupIndex++ {
		start := groupIndex * NativeGroupSize
		end := min(start+NativeGroupSize, len(values))
		groupValues := make([]float32, NativeGroupSize)
		copy(groupValues, values[start:end])

		encoded, err := EncodeVector(groupValues, preset)
		if err != nil {
			return NativeGroupedVector{}, err
		}

		payload, err := encoded.MarshalBinary()
		if err != nil {
			return NativeGroupedVector{}, err
		}

		logicalLen := end - start
		groups = append(groups, NativeGroupedGroup{
			Index:      groupIndex,
			LogicalLen: logicalLen,
			Scale:      blockScale(groupValues[:logicalLen]),
			Norm:       nativeGroupedVectorNorm(groupValues[:logicalLen]),
			Payload:    payload,
		})
	}

	return NativeGroupedVector{
		Header: header,
		Groups: groups,
		Preset: preset,
	}, nil
}

func DecodeNativeGroupedVector(v NativeGroupedVector) ([]float32, error) {
	if err := v.Validate(); err != nil {
		return nil, err
	}

	decoded := make([]float32, 0, v.Header.OriginalHeadDim)
	for groupIndex, group := range v.Groups {
		if group.Index != groupIndex {
			return nil, fmt.Errorf("native grouped payload index mismatch: got %d want %d", group.Index, groupIndex)
		}
		if group.LogicalLen <= 0 || group.LogicalLen > NativeGroupSize {
			return nil, fmt.Errorf("invalid native grouped logical length %d", group.LogicalLen)
		}

		groupValues, _, err := DecodeVector(group.Payload)
		if err != nil {
			return nil, err
		}
		if len(groupValues) != NativeGroupSize {
			return nil, fmt.Errorf("decoded native group size %d does not match %d", len(groupValues), NativeGroupSize)
		}

		decoded = append(decoded, groupValues[:group.LogicalLen]...)
	}

	if len(decoded) != v.Header.OriginalHeadDim {
		return nil, fmt.Errorf("decoded native grouped vector dim %d does not match header dim %d", len(decoded), v.Header.OriginalHeadDim)
	}

	return decoded, nil
}

func (v NativeGroupedVector) Validate() error {
	if err := v.Header.Validate(); err != nil {
		return err
	}
	if len(v.Groups) != v.Header.GroupCount {
		return fmt.Errorf("native grouped payload count %d does not match header group count %d", len(v.Groups), v.Header.GroupCount)
	}
	for i, group := range v.Groups {
		if group.Index != i {
			return fmt.Errorf("native grouped payload index mismatch: got %d want %d", group.Index, i)
		}
		if group.LogicalLen <= 0 || group.LogicalLen > NativeGroupSize {
			return fmt.Errorf("invalid native grouped logical length %d", group.LogicalLen)
		}
		if i < len(v.Groups)-1 && group.LogicalLen != NativeGroupSize {
			return fmt.Errorf("non-terminal native group %d has logical length %d", i, group.LogicalLen)
		}
		if i == len(v.Groups)-1 && group.LogicalLen+v.Header.TailPad != NativeGroupSize {
			return fmt.Errorf("tail padding %d does not match last group logical length %d", v.Header.TailPad, group.LogicalLen)
		}
		if len(group.Payload) == 0 {
			return fmt.Errorf("native grouped payload for group %d is empty", i)
		}
	}
	return nil
}

func (v NativeGroupedVector) MarshalBinary() ([]byte, error) {
	if err := v.Validate(); err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	buf.WriteString(nativeLayoutMagic)
	for _, field := range []any{
		uint8(v.Header.LayoutVersion),
		uint8(v.Preset.ID),
		uint32(v.Header.GroupSize),
		uint32(v.Header.GroupCount),
		uint32(v.Header.OriginalHeadDim),
		uint32(v.Header.PaddedDim),
		uint32(v.Header.TailPad),
		uint32(v.Header.GroupPayloadCount),
		uint32(v.Header.PerGroupScaleCount),
		uint32(v.Header.PerGroupNormCount),
	} {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			return nil, err
		}
	}
	if err := writeString(&buf, v.Header.LayoutKind); err != nil {
		return nil, err
	}
	if err := writeString(&buf, v.Header.VectorKind); err != nil {
		return nil, err
	}

	for _, group := range v.Groups {
		for _, field := range []any{
			uint32(group.Index),
			uint32(group.LogicalLen),
			group.Scale,
			group.Norm,
			uint32(len(group.Payload)),
		} {
			if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
				return nil, err
			}
		}
		buf.Write(group.Payload)
	}

	return buf.Bytes(), nil
}

func (v *NativeGroupedVector) UnmarshalBinary(data []byte) error {
	r := bytes.NewReader(data)
	magic := make([]byte, len(nativeLayoutMagic))
	if _, err := io.ReadFull(r, magic); err != nil {
		return err
	}
	if string(magic) != nativeLayoutMagic {
		return fmt.Errorf("invalid native grouped payload magic %q", string(magic))
	}

	var version uint8
	var presetID uint8
	var groupSize uint32
	var groupCount uint32
	var originalHeadDim uint32
	var paddedDim uint32
	var tailPad uint32
	var groupPayloadCount uint32
	var perGroupScaleCount uint32
	var perGroupNormCount uint32
	for _, field := range []any{
		&version,
		&presetID,
		&groupSize,
		&groupCount,
		&originalHeadDim,
		&paddedDim,
		&tailPad,
		&groupPayloadCount,
		&perGroupScaleCount,
		&perGroupNormCount,
	} {
		if err := binary.Read(r, binary.LittleEndian, field); err != nil {
			return err
		}
	}

	layoutKind, err := readString(r)
	if err != nil {
		return err
	}
	vectorKind, err := readString(r)
	if err != nil {
		return err
	}
	preset, err := PresetByID(presetID)
	if err != nil {
		return err
	}

	v.Header = NativeGroupedHeader{
		LayoutVersion:      int(version),
		LayoutKind:         layoutKind,
		GroupSize:          int(groupSize),
		GroupCount:         int(groupCount),
		OriginalHeadDim:    int(originalHeadDim),
		PaddedDim:          int(paddedDim),
		TailPad:            int(tailPad),
		GroupPayloadCount:  int(groupPayloadCount),
		PerGroupScaleCount: int(perGroupScaleCount),
		PerGroupNormCount:  int(perGroupNormCount),
		VectorKind:         vectorKind,
	}
	v.Preset = preset

	groups := make([]NativeGroupedGroup, 0, v.Header.GroupCount)
	for i := 0; i < v.Header.GroupCount; i++ {
		var index uint32
		var logicalLen uint32
		var scale float32
		var norm float32
		var payloadLen uint32
		for _, field := range []any{&index, &logicalLen, &scale, &norm, &payloadLen} {
			if err := binary.Read(r, binary.LittleEndian, field); err != nil {
				return err
			}
		}
		payload := make([]byte, payloadLen)
		if _, err := io.ReadFull(r, payload); err != nil {
			return err
		}
		groups = append(groups, NativeGroupedGroup{
			Index:      int(index),
			LogicalLen: int(logicalLen),
			Scale:      scale,
			Norm:       norm,
			Payload:    payload,
		})
	}
	if r.Len() != 0 {
		return fmt.Errorf("unexpected trailing bytes in native grouped payload: %d", r.Len())
	}

	v.Groups = groups
	return v.Validate()
}

func writeString(buf *bytes.Buffer, value string) error {
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(value))); err != nil {
		return err
	}
	_, err := buf.WriteString(value)
	return err
}

func readString(r *bytes.Reader) (string, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n == 0 {
		return "", nil
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func nativeGroupedVectorNorm(values []float32) float32 {
	if len(values) == 0 {
		return 0
	}
	var sumSquares float64
	for _, value := range values {
		sumSquares += float64(value * value)
	}
	if sumSquares < 1e-12 {
		return 0
	}
	return float32(math.Sqrt(sumSquares))
}
