package turboquant

import (
	"bytes"
	"math"
	"testing"
)

func TestNewNativeGroupedHeader(t *testing.T) {
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
	if h.GroupPayloadCount != 2 || h.PerGroupScaleCount != 2 || h.PerGroupNormCount != 2 {
		t.Fatalf("unexpected grouped counts: %+v", h)
	}
	if err := h.Validate(); err != nil {
		t.Fatalf("header should validate: %v", err)
	}
}

func TestNativeGroupedVectorRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		in   []float32
	}{
		{name: "short", in: []float32{1, 2, 3, 4}},
		{name: "exact128", in: makeRamp(128)},
		{name: "tailpad", in: makeRamp(130)},
		{name: "zeroish", in: make([]float32, 17)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded, err := EncodeNativeGroupedVector(tt.in, PresetTQ35)
			if err != nil {
				t.Fatalf("EncodeNativeGroupedVector: %v", err)
			}
			data, err := encoded.MarshalBinary()
			if err != nil {
				t.Fatalf("MarshalBinary: %v", err)
			}

			var decodedPayload NativeGroupedVector
			if err := decodedPayload.UnmarshalBinary(data); err != nil {
				t.Fatalf("UnmarshalBinary: %v", err)
			}
			if decodedPayload.Header.OriginalHeadDim != len(tt.in) {
				t.Fatalf("OriginalHeadDim = %d, want %d", decodedPayload.Header.OriginalHeadDim, len(tt.in))
			}
			if decodedPayload.Header.LayoutKind != NativeLayoutKind128 {
				t.Fatalf("LayoutKind = %q, want %q", decodedPayload.Header.LayoutKind, NativeLayoutKind128)
			}

			decoded, err := DecodeNativeGroupedVector(decodedPayload)
			if err != nil {
				t.Fatalf("DecodeNativeGroupedVector: %v", err)
			}
			if len(decoded) != len(tt.in) {
				t.Fatalf("decoded len = %d, want %d", len(decoded), len(tt.in))
			}
			if mse(tt.in, decoded) > 250 {
				t.Fatalf("round-trip mse = %v, want <= 250", mse(tt.in, decoded))
			}
		})
	}
}

func TestNativeGroupedVectorRejectsMalformedPayloads(t *testing.T) {
	valid, err := EncodeNativeGroupedVector(makeRamp(130), PresetTQ25)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("bad version", func(t *testing.T) {
		bad := valid
		bad.Header.LayoutVersion++
		if err := bad.Validate(); err == nil {
			t.Fatal("expected invalid version to fail")
		}
	})

	t.Run("bad kind", func(t *testing.T) {
		bad := valid
		bad.Header.LayoutKind = ReferenceLayoutKind
		if err := bad.Validate(); err == nil {
			t.Fatal("expected invalid layout kind to fail")
		}
	})

	t.Run("bad tail pad", func(t *testing.T) {
		bad := valid
		bad.Header.TailPad++
		if err := bad.Validate(); err == nil {
			t.Fatal("expected invalid tail pad to fail")
		}
	})

	t.Run("payload count mismatch", func(t *testing.T) {
		bad := valid
		bad.Groups = bad.Groups[:1]
		if err := bad.Validate(); err == nil {
			t.Fatal("expected payload count mismatch to fail")
		}
	})

	t.Run("truncated payload", func(t *testing.T) {
		data, err := valid.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		var decoded NativeGroupedVector
		if err := decoded.UnmarshalBinary(data[:len(data)-4]); err == nil {
			t.Fatal("expected truncated payload to fail")
		}
	})

	t.Run("reference payload rejected", func(t *testing.T) {
		ref, err := EncodeVector(makeRamp(128), PresetTQ35)
		if err != nil {
			t.Fatal(err)
		}
		data, err := ref.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		var decoded NativeGroupedVector
		if err := decoded.UnmarshalBinary(data); err == nil {
			t.Fatal("expected reference payload to fail grouped unmarshal")
		}
	})
}

func TestNativeGroupedMarshalIsDeterministic(t *testing.T) {
	v, err := EncodeNativeGroupedVector(makeRamp(129), PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	dataA, err := v.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	dataB, err := v.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(dataA, dataB) {
		t.Fatal("marshal output is not deterministic")
	}
}

func TestExperimentalSegmentedHeadDimPlan(t *testing.T) {
	plan, ok := ExperimentalSegmentedHeadDimPlan(576)
	if !ok {
		t.Fatal("expected segmented plan for head_dim=576")
	}
	if !plan.Experimental {
		t.Fatal("expected segmented plan to remain experimental")
	}
	if len(plan.Segments) != 3 || plan.Segments[0] != 256 || plan.Segments[1] != 256 || plan.Segments[2] != 64 {
		t.Fatalf("unexpected segmented plan: %+v", plan)
	}
	if _, ok := ExperimentalSegmentedHeadDimPlan(512); ok {
		t.Fatal("did not expect segmented plan for power-of-two head dim")
	}
}

func TestExperimentalSegmentedHeadVectorRoundTrip(t *testing.T) {
	plan, ok := ExperimentalSegmentedHeadDimPlan(576)
	if !ok {
		t.Fatal("expected segmented plan for head_dim=576")
	}
	in := makeRamp(576)
	encoded, err := EncodeExperimentalSegmentedHeadVector(in, PresetTQ35, plan)
	if err != nil {
		t.Fatalf("EncodeExperimentalSegmentedHeadVector: %v", err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	var decodedPayload NativeSegmentedHeadVector
	if err := decodedPayload.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary: %v", err)
	}
	out, err := DecodeExperimentalSegmentedHeadVector(decodedPayload)
	if err != nil {
		t.Fatalf("DecodeExperimentalSegmentedHeadVector: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("decoded len = %d, want %d", len(out), len(in))
	}
	if mse(in, out) > 5000 {
		t.Fatalf("segmented round-trip mse = %v, want <= 5000", mse(in, out))
	}
}

func makeRamp(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i + 1)
	}
	return out
}

func mse(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return math.Inf(1)
	}
	var sum float64
	for i := range a {
		delta := float64(a[i] - b[i])
		sum += delta * delta
	}
	return sum / float64(len(a))
}
