package ollamarunner

import (
	"iter"
	"testing"

	"github.com/ollama/ollama/kvcache"
)

type stubConfig struct {
	arch string
	u32  map[string]uint32
	i32s map[string][]int32
	any  map[string]any
}

func (s stubConfig) Architecture() string            { return s.arch }
func (s stubConfig) String(string, ...string) string { return "" }
func (s stubConfig) Uint(key string, fallback ...uint32) uint32 {
	if v, ok := s.u32[key]; ok {
		return v
	}
	if len(fallback) > 0 {
		return fallback[0]
	}
	return 0
}
func (s stubConfig) Float(string, ...float32) float32 { return 0 }
func (s stubConfig) Bool(string, ...bool) bool        { return false }
func (s stubConfig) Strings(_ string, _ ...[]string) []string {
	return nil
}
func (s stubConfig) Ints(key string, _ ...[]int32) []int32 {
	if v, ok := s.i32s[key]; ok {
		return append([]int32(nil), v...)
	}
	return nil
}
func (s stubConfig) Floats(_ string, _ ...[]float32) []float32 { return nil }
func (s stubConfig) Bools(_ string, _ ...[]bool) []bool        { return nil }
func (s stubConfig) Len() int {
	return len(s.u32) + len(s.i32s) + len(s.any)
}
func (s stubConfig) Keys() iter.Seq[string] {
	return func(yield func(string) bool) {
		for k := range s.u32 {
			if !yield(k) {
				return
			}
		}
		for k := range s.i32s {
			if !yield(k) {
				return
			}
		}
		for k := range s.any {
			if !yield(k) {
				return
			}
		}
	}
}
func (s stubConfig) Value(key string) any {
	if v, ok := s.any[key]; ok {
		return v
	}
	if v, ok := s.i32s[key]; ok {
		return v
	}
	if v, ok := s.u32[key]; ok {
		return v
	}
	return nil
}

func TestDetectHeadDimPrefersKeyLength(t *testing.T) {
	cfg := stubConfig{
		arch: "llama",
		u32: map[string]uint32{
			"attention.key_length":    192,
			"attention.value_length":  128,
			"embedding_length":        4096,
			"attention.head_count_kv": 32,
			"attention.head_count":    16,
			"rope.dimension_count":    64,
		},
	}

	got, source := detectHeadDim(cfg)
	if got != 192 || source != turboQuantHeadDimSourceAttentionKeyLength {
		t.Fatalf("detectHeadDim() = (%d, %s), want (192, %s)", got, source, turboQuantHeadDimSourceAttentionKeyLength)
	}
}

func TestDetectHeadDimFallsBackToKVHeads(t *testing.T) {
	cfg := stubConfig{
		arch: "llama",
		u32: map[string]uint32{
			"embedding_length":        4096,
			"attention.head_count_kv": 32,
		},
	}

	got, source := detectHeadDim(cfg)
	if got != 128 || source != turboQuantHeadDimSourceEmbeddingHeadCountKV {
		t.Fatalf("detectHeadDim() = (%d, %s), want (128, %s)", got, source, turboQuantHeadDimSourceEmbeddingHeadCountKV)
	}
}

func TestDetectHeadDimFallsBackToHeadCount(t *testing.T) {
	cfg := stubConfig{
		arch: "qwen3",
		u32: map[string]uint32{
			"embedding_length":     4096,
			"attention.head_count": 64,
		},
	}

	got, source := detectHeadDim(cfg)
	if got != 64 || source != turboQuantHeadDimSourceEmbeddingHeadCount {
		t.Fatalf("detectHeadDim() = (%d, %s), want (64, %s)", got, source, turboQuantHeadDimSourceEmbeddingHeadCount)
	}
}

func TestDetectHeadDimUsesUniformLayerArray(t *testing.T) {
	cfg := stubConfig{
		arch: "gemma2",
		u32: map[string]uint32{
			"embedding_length": 4096,
		},
		i32s: map[string][]int32{
			"attention.head_count": {64, 64, 64},
		},
	}

	got, source := detectHeadDim(cfg)
	if got != 64 || source != turboQuantHeadDimSourceLayerHeadCountArray {
		t.Fatalf("detectHeadDim() = (%d, %s), want (64, %s)", got, source, turboQuantHeadDimSourceLayerHeadCountArray)
	}
}

func TestDetectHeadDimFallsBackToRopeDimensionCount(t *testing.T) {
	cfg := stubConfig{
		arch: "deepseek2",
		u32: map[string]uint32{
			"rope.dimension_count": 128,
		},
	}

	got, source := detectHeadDim(cfg)
	if got != 128 || source != turboQuantHeadDimSourceRopeDimensionCount {
		t.Fatalf("detectHeadDim() = (%d, %s), want (128, %s)", got, source, turboQuantHeadDimSourceRopeDimensionCount)
	}
}

func TestDetectTurboQuantModelSupportTiers(t *testing.T) {
	tests := []struct {
		name       string
		cfg        stubConfig
		wantClass  string
		wantTier   turboQuantSupportTier
		wantNative bool
	}{
		{
			name:       "safe",
			cfg:        stubConfig{arch: "llama", u32: map[string]uint32{"attention.key_length": 128}},
			wantClass:  "llama-family",
			wantTier:   turboQuantSupportSafe,
			wantNative: true,
		},
		{
			name:       "conservative",
			cfg:        stubConfig{arch: "qwen3", u32: map[string]uint32{"attention.key_length": 64}},
			wantClass:  "qwen-family",
			wantTier:   turboQuantSupportConservative,
			wantNative: true,
		},
		{
			name:       "experimental",
			cfg:        stubConfig{arch: "llama", u32: map[string]uint32{"attention.key_length": 256}},
			wantClass:  "llama-family",
			wantTier:   turboQuantSupportExperimental,
			wantNative: true,
		},
		{
			name:       "unsupported",
			cfg:        stubConfig{arch: "llama", u32: map[string]uint32{"attention.key_length": 160}},
			wantClass:  "llama-family",
			wantTier:   turboQuantSupportConservative,
			wantNative: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := newRunnerTestModel(kvcache.NewCausalCache(nil), &runnerTestBackend{config: tt.cfg})
			got := detectTurboQuantModelSupport(model)
			if got.ArchitectureClass != tt.wantClass {
				t.Fatalf("ArchitectureClass = %q, want %q", got.ArchitectureClass, tt.wantClass)
			}
			if got.SupportTier != tt.wantTier {
				t.Fatalf("SupportTier = %q, want %q", got.SupportTier, tt.wantTier)
			}
			if got.NativeTurboQuantAllowed != tt.wantNative {
				t.Fatalf("NativeTurboQuantAllowed = %t, want %t", got.NativeTurboQuantAllowed, tt.wantNative)
			}
		})
	}
}

func TestDetectTurboQuantModelSupportRejectsNonAlignedHeadDim(t *testing.T) {
	model := newRunnerTestModel(kvcache.NewCausalCache(nil), &runnerTestBackend{
		config: stubConfig{arch: "llama", u32: map[string]uint32{"attention.key_length": 130}},
	})

	got := detectTurboQuantModelSupport(model)
	if got.SupportTier != turboQuantSupportUnsupported {
		t.Fatalf("SupportTier = %q, want unsupported", got.SupportTier)
	}
	if got.NativeTurboQuantAllowed {
		t.Fatalf("NativeTurboQuantAllowed = true, want false")
	}
	if got.UnsupportedReason == "" {
		t.Fatal("expected unsupported reason")
	}
}

func TestDetectTurboQuantModelSupportMarksHybridWrapperPaths(t *testing.T) {
	model := newRunnerTestModel(
		kvcache.NewWrapperCache(kvcache.NewEncoderCache(), kvcache.NewCausalCache(nil)),
		&runnerTestBackend{config: stubConfig{arch: "gptoss", u32: map[string]uint32{"attention.key_length": 128}}},
	)

	got := detectTurboQuantModelSupport(model)
	if !got.HybridKVArchitecture {
		t.Fatal("expected hybrid architecture")
	}
	if got.ArchitectureClass != "gptoss-hybrid" {
		t.Fatalf("ArchitectureClass = %q, want gptoss-hybrid", got.ArchitectureClass)
	}
	if got.NativeTurboQuantAllowed {
		t.Fatal("expected native rollout to be disabled for hybrid architecture")
	}
}

func TestDetectTurboQuantModelSupportMarksMLAUnsupported(t *testing.T) {
	model := newRunnerTestModel(kvcache.NewCausalCache(nil), &runnerTestBackend{
		config: stubConfig{arch: "deepseek2", u32: map[string]uint32{"rope.dimension_count": 128}},
	})

	got := detectTurboQuantModelSupport(model)
	if got.ArchitectureClass != "deepseek-mla" {
		t.Fatalf("ArchitectureClass = %q, want deepseek-mla", got.ArchitectureClass)
	}
	if got.SupportTier != turboQuantSupportUnsupported {
		t.Fatalf("SupportTier = %q, want unsupported", got.SupportTier)
	}
	if got.NativeTurboQuantAllowed {
		t.Fatal("expected native rollout disabled")
	}
}
