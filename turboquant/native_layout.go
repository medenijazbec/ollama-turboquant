package turboquant

import "fmt"

const (
	// @TheTom: native TurboQuant storage should use 128-element groups, not whole-vector blocks.
	// @signalnine: block size 128 is the current native CUDA direction.
	// @Madreag: CUDA turbo3 path uses 128-element grouping with 4x32 internal structure.
	NativeGroupSize     = 128
	NativeLayoutVersion = 1
	NativeLayoutKind128 = "native_grouped_128"
)

type NativeGroupedHeader struct {
	LayoutVersion   int
	LayoutKind      string
	GroupSize       int
	GroupCount      int
	OriginalHeadDim int
	PaddedDim       int
	TailPad         int
}

func NewNativeGroupedHeader(headDim int) NativeGroupedHeader {
	groupCount := 0
	if headDim > 0 {
		groupCount = (headDim + NativeGroupSize - 1) / NativeGroupSize
	}
	paddedDim := groupCount * NativeGroupSize
	return NativeGroupedHeader{
		LayoutVersion:   NativeLayoutVersion,
		LayoutKind:      NativeLayoutKind128,
		GroupSize:       NativeGroupSize,
		GroupCount:      groupCount,
		OriginalHeadDim: headDim,
		PaddedDim:       paddedDim,
		TailPad:         paddedDim - headDim,
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
	if h.GroupCount < 0 || h.OriginalHeadDim < 0 || h.PaddedDim < 0 || h.TailPad < 0 {
		return fmt.Errorf("invalid native grouped header values")
	}
	if h.GroupCount*NativeGroupSize != h.PaddedDim {
		return fmt.Errorf("padded dim %d does not match group count %d", h.PaddedDim, h.GroupCount)
	}
	if h.OriginalHeadDim+h.TailPad != h.PaddedDim {
		return fmt.Errorf("tail padding %d does not match original head dim %d", h.TailPad, h.OriginalHeadDim)
	}
	return nil
}
