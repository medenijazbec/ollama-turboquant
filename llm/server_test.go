package llm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"golang.org/x/sync/semaphore"
)

func TestLLMServerFitGPU(t *testing.T) {
	minMemory := 457 * format.MebiByte

	tests := []struct {
		name        string
		gpus        []ml.DeviceInfo
		layers      []int
		numGPU      int
		requireFull bool
		expected    ml.GPULayersList
		expectedErr error
	}{
		{
			name:        "No GPU",
			layers:      []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:      -1,
			expected:    ml.GPULayersList{},
			requireFull: true, // Should not try to evict even though we can't load any layers
		},
		{
			name:     "Full single GPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1, 2}}},
		},
		{
			name:     "Partial single GPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1, 2}}},
		},
		{
			name:     "Single GPU with numGPU 1",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1}}},
		},
		{
			name:     "Single GPU with numGPU 0",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   0,
			expected: ml.GPULayersList{},
		},
		{
			name:     "Single GPU with numGPU 999",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1, 2, 3}}},
		},
		{
			name:     "Multi GPU fits on one",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0, 1, 2}}},
		},
		{
			name:     "Multi GPU split",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1, 2}}},
		},
		{
			name:     "Multi GPU partial",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 256 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 1",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 2",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   2,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1}}},
		},
		{
			name:     "Multi GPU numGPU 999",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{256 * format.MebiByte, 256 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0, 1}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{2}}},
		},
		{
			name:     "Multi GPU different libraries",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{Library: "CUDA", ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{Library: "ROCm", ID: "gpu1"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{128 * format.MebiByte, 128 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1", Library: "ROCm"}, Layers: []int{0, 1}}},
		},
		{
			name:        "requireFull",
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:      []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:      -1,
			requireFull: true,
			expectedErr: ErrLoadRequiredFull,
		},
		{
			name:        "requireFull numGPU",
			gpus:        []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(256 * format.MebiByte)}},
			layers:      []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:      4,
			requireFull: true,
			expectedErr: ErrLoadRequiredFull,
		},
		{
			name:     "iGPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1, 2}}},
		},
		{
			name:     "iGPU + dGPU",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{1, 2}}},
		},
		{
			name:     "iGPU + dGPU fits on one",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{50 * format.MebiByte, 50 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0, 1}}},
		},
		{
			name:     "iGPU + dGPU partial",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   -1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{0, 1}}, {DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{2}}},
		},
		{
			name:     "iGPU + dGPU numGPU 1",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   1,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{2}}},
		},
		{
			name:     "iGPU + dGPU numGPU 999",
			gpus:     []ml.DeviceInfo{{DeviceID: ml.DeviceID{ID: "gpu0"}, FreeMemory: uint64(128*format.MebiByte + minMemory)}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Integrated: true, FreeMemory: uint64(256*format.MebiByte + minMemory)}},
			layers:   []int{100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte, 100 * format.MebiByte},
			numGPU:   999,
			expected: ml.GPULayersList{{DeviceID: ml.DeviceID{ID: "gpu0"}, Layers: []int{0}}, {DeviceID: ml.DeviceID{ID: "gpu1"}, Layers: []int{1, 2, 3}}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var systemInfo ml.SystemInfo
			systemInfo.TotalMemory = format.GibiByte
			systemInfo.FreeMemory = 512 * format.MebiByte
			systemInfo.FreeSwap = 256 * format.MebiByte

			s := &ollamaServer{
				llmServer: llmServer{
					totalLayers: uint64(len(tt.layers)),
					options: api.Options{
						Runner: api.Runner{
							NumGPU: tt.numGPU,
						},
					},
				},
			}

			s.mem = &ml.BackendMemory{CPU: ml.DeviceMemory{
				Weights: make([]uint64, s.totalLayers),
				Cache:   make([]uint64, s.totalLayers),
			}, GPUs: make([]ml.DeviceMemory, len(tt.gpus))}

			for i := range tt.layers {
				s.mem.CPU.Weights[i] = uint64(tt.layers[i])
			}

			for i := range s.mem.GPUs {
				s.mem.GPUs[i].DeviceID = tt.gpus[i].DeviceID
				s.mem.GPUs[i].Weights = make([]uint64, s.totalLayers)
				s.mem.GPUs[i].Cache = make([]uint64, s.totalLayers)
			}

			gpuLayers, err := s.createLayout(systemInfo, tt.gpus, s.mem, tt.requireFull, 0)
			if err != tt.expectedErr {
				t.Fatalf("fitGPU returned error: %v", err)
			}
			if gpuLayers.Hash() != tt.expected.Hash() {
				t.Errorf("fitGPU assigned %v, want %v", gpuLayers, tt.expected)
			}
		})
	}
}

func TestLLMServerCompletionFormat(t *testing.T) {
	// This test was written to fix an already deployed issue. It is a bit
	// of a mess, and but it's good enough, until we can refactoring the
	// Completion method to be more testable.

	ctx, cancel := context.WithCancel(t.Context())
	s := &llmServer{
		sem: semaphore.NewWeighted(1), // required to prevent nil panic
	}

	checkInvalid := func(format string) {
		t.Helper()
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(format),
		}, nil)

		want := fmt.Sprintf("invalid format: %q; expected \"json\" or a valid JSON Schema", format)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("err = %v; want %q", err, want)
		}
	}

	checkInvalid("X")   // invalid format
	checkInvalid(`"X"`) // invalid JSON Schema

	cancel() // prevent further processing if request makes it past the format check

	checkValid := func(err error) {
		t.Helper()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Completion: err = %v; expected context.Canceled", err)
		}
	}

	valids := []string{
		// "missing"
		``,
		`""`,
		`null`,

		// JSON
		`"json"`,
		`{"type":"object"}`,
	}
	for _, valid := range valids {
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(valid),
		}, nil)
		checkValid(err)
	}

	err := s.Completion(ctx, CompletionRequest{
		Options: new(api.Options),
		Format:  nil, // missing format
	}, nil)
	checkValid(err)
}

func TestNewKVCacheMode(t *testing.T) {
	tests := []struct {
		in        string
		requested string
		effective string
		aliased   bool
	}{
		{in: "", requested: "", effective: "", aliased: false},
		{in: "f16", requested: "f16", effective: "f16", aliased: false},
		{in: "TQ25", requested: "tq25", effective: "tq25", aliased: false},
		{in: "tq3", requested: "tq3", effective: "tq35", aliased: true},
		{in: "Tq4", requested: "tq4", effective: "tq35", aliased: true},
	}

	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			got := newKVCacheMode(tt.in)
			if got.Requested != tt.requested || got.Effective != tt.effective || got.Aliased != tt.aliased {
				t.Fatalf("newKVCacheMode(%q) = %+v, want requested=%q effective=%q aliased=%v", tt.in, got, tt.requested, tt.effective, tt.aliased)
			}
		})
	}
}

func TestResolveKVCacheMode(t *testing.T) {
	t.Run("request option overrides env", func(t *testing.T) {
		t.Setenv("OLLAMA_KV_CACHE_TYPE", "q8_0")
		got := resolveKVCacheMode(api.Options{Runner: api.Runner{KVCacheType: "tq3"}})
		if got.Requested != "tq3" || got.Effective != "tq35" {
			t.Fatalf("resolveKVCacheMode() = %+v, want requested=tq3 effective=tq35", got)
		}
	})

	t.Run("env used when request unset", func(t *testing.T) {
		t.Setenv("OLLAMA_KV_CACHE_TYPE", "tq25")
		got := resolveKVCacheMode(api.Options{})
		if got.Requested != "tq25" || got.Effective != "tq25" {
			t.Fatalf("resolveKVCacheMode() = %+v, want requested=tq25 effective=tq25", got)
		}
	})
}

func TestResolveKVCacheModes(t *testing.T) {
	t.Run("request split overrides unified request", func(t *testing.T) {
		got := resolveKVCacheModes(api.Options{
			Runner: api.Runner{
				KVCacheType:  "tq35",
				KVCacheTypeK: "q8_0",
			},
		})

		if got.Unified.Effective != "tq35" {
			t.Fatalf("unified effective = %q, want tq35", got.Unified.Effective)
		}
		if got.K.Effective != "q8_0" {
			t.Fatalf("K effective = %q, want q8_0", got.K.Effective)
		}
		if got.V.Effective != "tq35" {
			t.Fatalf("V effective = %q, want tq35", got.V.Effective)
		}
		if !got.Asymmetric || got.Symmetric {
			t.Fatalf("modes = %+v, want asymmetric split resolution", got)
		}
	})

	t.Run("env split overrides env unified", func(t *testing.T) {
		t.Setenv("OLLAMA_KV_CACHE_TYPE", "tq35")
		t.Setenv("OLLAMA_KV_CACHE_TYPE_K", "q8_0")
		t.Setenv("OLLAMA_KV_CACHE_TYPE_V", "tq25")

		got := resolveKVCacheModes(api.Options{})

		if got.K.Effective != "q8_0" || got.V.Effective != "tq25" {
			t.Fatalf("resolveKVCacheModes() = %+v, want q8_0/tq25", got)
		}
		if !got.Asymmetric {
			t.Fatalf("modes = %+v, want asymmetric=true", got)
		}
	})

	t.Run("unset defaults to f16 symmetric", func(t *testing.T) {
		got := resolveKVCacheModes(api.Options{})
		if got.K.Effective != "f16" || got.V.Effective != "f16" {
			t.Fatalf("resolveKVCacheModes() = %+v, want f16/f16", got)
		}
		if !got.Symmetric || got.Asymmetric {
			t.Fatalf("modes = %+v, want symmetric=true asymmetric=false", got)
		}
	})
}

func TestLogKVCacheModeOverrides(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn}))
	prev := slog.Default()
	slog.SetDefault(logger)
	defer slog.SetDefault(prev)

	logKVCacheModeOverrides(kvCacheModes{
		Unified: newKVCacheMode("tq35"),
		K:       newKVCacheMode("q8_0"),
		V:       newKVCacheMode("tq35"),
	})

	out := buf.String()
	if !strings.Contains(out, "split kv cache settings override unified kv_cache_type") {
		t.Fatalf("log output missing override warning: %q", out)
	}
	if !strings.Contains(out, "override_sides=K") {
		t.Fatalf("log output missing override side: %q", out)
	}
}

func TestResolveKVCacheBackendMode(t *testing.T) {
	t.Run("request option overrides env", func(t *testing.T) {
		t.Setenv("OLLAMA_KV_CACHE_BACKEND", "")
		got := resolveKVCacheBackendMode(api.Options{Runner: api.Runner{KVCacheBackend: "cuda"}})
		if got.Requested != "cuda" || got.Effective != "cuda" {
			t.Fatalf("resolveKVCacheBackendMode() = %+v, want requested=cuda effective=cuda", got)
		}
	})

	t.Run("invalid env falls back empty", func(t *testing.T) {
		t.Setenv("OLLAMA_KV_CACHE_BACKEND", "metal")
		got := resolveKVCacheBackendMode(api.Options{})
		if got.Requested != "metal" || got.Effective != "" {
			t.Fatalf("resolveKVCacheBackendMode() = %+v, want requested=metal effective=''", got)
		}
	})
}

func TestSelectLegacyKVCacheType(t *testing.T) {
	tests := []struct {
		name           string
		mode           kvCacheMode
		flashAttention ml.FlashAttentionType
		supported      bool
		quantized      bool
		assigned       string
		warning        string
	}{
		{
			name:           "AliasAccepted",
			mode:           newKVCacheMode("tq3"),
			flashAttention: ml.FlashAttentionEnabled,
			supported:      true,
			quantized:      true,
			assigned:       "tq35",
		},
		{
			name:           "QuantizedRequiresFlashAttention",
			mode:           newKVCacheMode("tq25"),
			flashAttention: ml.FlashAttentionDisabled,
			supported:      true,
			quantized:      true,
			warning:        "OLLAMA_FLASH_ATTENTION must be enabled to use a quantized OLLAMA_KV_CACHE_TYPE",
		},
		{
			name:           "Unsupported",
			mode:           newKVCacheMode("turbo"),
			flashAttention: ml.FlashAttentionEnabled,
			supported:      false,
			quantized:      false,
			warning:        "unsupported OLLAMA_KV_CACHE_TYPE",
		},
		{
			name:           "CanonicalF16",
			mode:           newKVCacheMode("f16"),
			flashAttention: ml.FlashAttentionDisabled,
			supported:      true,
			quantized:      false,
			assigned:       "f16",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := selectLegacyKVCacheType(tt.mode, tt.flashAttention, func(string) bool { return tt.supported }, func(string) bool { return tt.quantized })
			if got.Assigned != tt.assigned || got.Warning != tt.warning {
				t.Fatalf("selectLegacyKVCacheType() = %+v, want assigned=%q warning=%q", got, tt.assigned, tt.warning)
			}
		})
	}
}

func TestSelectEngineKVCacheType(t *testing.T) {
	tests := []struct {
		name                  string
		mode                  kvCacheMode
		flashAttentionEnabled bool
		supported             bool
		assigned              string
		warning               string
	}{
		{
			name:                  "AliasAccepted",
			mode:                  newKVCacheMode("tq4"),
			flashAttentionEnabled: true,
			supported:             true,
			assigned:              "tq35",
		},
		{
			name:                  "QuantizedRejectedWithoutFlashAttention",
			mode:                  newKVCacheMode("tq25"),
			flashAttentionEnabled: false,
			supported:             true,
			warning:               "quantized kv cache requested but flash attention disabled",
		},
		{
			name:                  "Unsupported",
			mode:                  newKVCacheMode("invalid"),
			flashAttentionEnabled: true,
			supported:             false,
			warning:               "kv cache type not supported by model",
		},
		{
			name:                  "CanonicalF16WithoutFlashAttention",
			mode:                  newKVCacheMode("f16"),
			flashAttentionEnabled: false,
			supported:             true,
			assigned:              "f16",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := selectEngineKVCacheType(tt.mode, tt.flashAttentionEnabled, func(string) bool { return tt.supported })
			if got.Assigned != tt.assigned || got.Warning != tt.warning {
				t.Fatalf("selectEngineKVCacheType() = %+v, want assigned=%q warning=%q", got, tt.assigned, tt.warning)
			}
		})
	}
}

func TestKVCacheModeLogAttrs(t *testing.T) {
	if got := kvCacheModeLogAttrs(newKVCacheMode("tq35")); fmt.Sprint(got) != "[effective tq35]" {
		t.Fatalf("canonical attrs = %v, want [effective tq35]", got)
	}

	if got := kvCacheModeLogAttrs(newKVCacheMode("tq3")); fmt.Sprint(got) != "[requested tq3 effective tq35]" {
		t.Fatalf("alias attrs = %v, want [requested tq3 effective tq35]", got)
	}
}

func TestLogAcceptedKVCacheType(t *testing.T) {
	var buf bytes.Buffer
	logger := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelInfo}))
	prev := slog.Default()
	slog.SetDefault(logger)
	defer slog.SetDefault(prev)

	logAcceptedKVCacheType(newKVCacheMode("tq3"))

	out := buf.String()
	if !strings.Contains(out, "using kv cache type") {
		t.Fatalf("log output missing message: %q", out)
	}
	if !strings.Contains(out, "requested=tq3") || !strings.Contains(out, "effective=tq35") {
		t.Fatalf("log output missing normalized attrs: %q", out)
	}
}
