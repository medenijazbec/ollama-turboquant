package turboquant

import "testing"

func TestNewNativeGroupedHeader(t *testing.T) {
	// @TheTom: native TurboQuant storage should use 128-element groups, not whole-vector blocks.
	// @signalnine: block size 128 is the current native CUDA direction.
	// @Madreag: CUDA turbo3 path uses 128-element grouping with 4x32 internal structure.
	h := NewNativeGroupedHeader(130)
	if h.GroupSize != NativeGroupSize {
		t.Fatalf("unexpected group size: %d", h.GroupSize)
	}
	if h.GroupCount != 2 {
		t.Fatalf("unexpected group count: %d", h.GroupCount)
	}
	if h.PaddedDim != 256 || h.TailPad != 126 {
		t.Fatalf("unexpected padding: padded=%d tail=%d", h.PaddedDim, h.TailPad)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("header should validate: %v", err)
	}
}

func TestNativeGroupedHeaderValidateRejectsMismatch(t *testing.T) {
	h := NewNativeGroupedHeader(128)
	h.PaddedDim++
	if err := h.Validate(); err == nil {
		t.Fatal("expected invalid header to fail validation")
	}
}
