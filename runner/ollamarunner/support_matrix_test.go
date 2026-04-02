package ollamarunner

import (
	"iter"
	"testing"
)

type stubConfig struct {
	arch string
	u32  map[string]uint32
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
func (s stubConfig) Float(string, ...float32) float32      { return 0 }
func (s stubConfig) Bool(string, ...bool) bool             { return false }
func (s stubConfig) Strings(string, ...[]string) []string  { return nil }
func (s stubConfig) Ints(string, ...[]int32) []int32       { return nil }
func (s stubConfig) Floats(string, ...[]float32) []float32 { return nil }
func (s stubConfig) Bools(string, ...[]bool) []bool        { return nil }
func (s stubConfig) Len() int                              { return len(s.u32) }
func (s stubConfig) Keys() iter.Seq[string] {
	return func(yield func(string) bool) {
		for k := range s.u32 {
			if !yield(k) {
				return
			}
		}
	}
}
func (s stubConfig) Value(key string) any { return s.u32[key] }

func TestDetectHeadDim(t *testing.T) {
	cfg := stubConfig{
		arch: "llama",
		u32: map[string]uint32{
			"llama.embedding_length":     4096,
			"llama.attention.head_count": 32,
		},
	}

	if got := detectHeadDim(cfg); got != 128 {
		t.Fatalf("detectHeadDim() = %d, want 128", got)
	}
}
