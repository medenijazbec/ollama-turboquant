package ollamarunner

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"reflect"
	"slices"
	"testing"
	"time"
	"unsafe"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

func TestCountCommon(t *testing.T) {
	tests := []struct {
		name     string
		t1       []*input.Input
		t2       []*input.Input
		expected int32
	}{
		{
			name:     "Equal",
			t1:       []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}},
			t2:       []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}},
			expected: 3,
		},
		{
			name:     "Prefix",
			t1:       []*input.Input{{Token: 1}},
			t2:       []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}},
			expected: 1,
		},
		{
			name:     "Image Prefix",
			t1:       []*input.Input{{MultimodalHash: 1}},
			t2:       []*input.Input{{MultimodalHash: 1}, {MultimodalHash: 2}, {MultimodalHash: 3}},
			expected: 1,
		},
		{
			name:     "Mixed",
			t1:       []*input.Input{{Token: 1}, {MultimodalHash: 1}},
			t2:       []*input.Input{{Token: 1}, {MultimodalHash: 1}, {Token: 5}},
			expected: 2,
		},
		{
			name:     "Mixed, Same Length",
			t1:       []*input.Input{{Token: 1}, {MultimodalHash: 1}},
			t2:       []*input.Input{{Token: 1}, {MultimodalHash: 2}},
			expected: 1,
		},
		{
			name:     "Empty",
			t1:       []*input.Input{},
			t2:       []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}},
			expected: 0,
		},
		{
			name:     "Both Empty",
			t1:       []*input.Input{},
			t2:       []*input.Input{},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := countCommonPrefix(tt.t1, tt.t2)
			if result != tt.expected {
				t.Errorf("countCommonPrefix(%v, %v): have %v; want %v", tt.t1, tt.t2, result, tt.expected)
			}
		})
	}
}

func TestKVCacheTypeFromStrTurboQuant(t *testing.T) {
	tests := map[string]ml.DType{
		"":     ml.DTypeF16,
		"f16":  ml.DTypeF16,
		"q8_0": ml.DTypeQ80,
		"q4_0": ml.DTypeQ40,
		"tq25": ml.DTypeTQ25,
		"tq35": ml.DTypeTQ35,
		"tq3":  ml.DTypeTQ35,
		"tq4":  ml.DTypeTQ35,
		"TQ35": ml.DTypeTQ35,
		"Tq3":  ml.DTypeTQ35,
	}

	for in, want := range tests {
		if got := kvCacheTypeFromStr(in); got != want {
			t.Fatalf("kvCacheTypeFromStr(%q) = %v, want %v", in, got, want)
		}
	}
}

func TestResolveKVCacheTypes(t *testing.T) {
	k, v := resolveKVCacheTypes("tq35", "q8_0", "")
	if k != "q8_0" || v != "tq35" {
		t.Fatalf("resolveKVCacheTypes override failed: k=%q v=%q", k, v)
	}

	k, v = resolveKVCacheTypes("", "", "off")
	if k != "f16" || v != "f16" {
		t.Fatalf("resolveKVCacheTypes default/off failed: k=%q v=%q", k, v)
	}
}

func TestKVCacheRuntimeInfoAsymmetric(t *testing.T) {
	info := (&InputCache{
		kvCacheRequested:          "tq35",
		kvCacheEffective:          "mixed",
		kvCacheRequestedK:         "q8_0",
		kvCacheRequestedV:         "tq35",
		kvCacheEffectiveK:         "q8_0",
		kvCacheEffectiveV:         "f16",
		kvAlgoResolved:            "mixed",
		kvAlgoResolvedK:           "",
		kvAlgoResolvedV:           "",
		kvCacheBackend:            "cuda",
		kvCachePath:               "mixed",
		kvCachePathK:              "dense-fallback",
		kvCachePathV:              "dense-fallback",
		kvSymmetric:               false,
		kvAsymmetric:              true,
		fallbackReason:            "requested V turboquant path is not supported",
		turboQuantPathKind:        "reference_wrapper",
		referenceTurboQuantActive: true,
		tqBlockSize:               0,
		tqLayoutKind:              turboquant.ReferenceLayoutKind,
		tqLayoutVersion:           turboquant.BlockVersion,
	}).RuntimeInfo()

	if !info.Asymmetric || info.Symmetric {
		t.Fatalf("unexpected symmetry flags: %+v", info)
	}
	if info.EffectiveK != "q8_0" || info.EffectiveV != "f16" {
		t.Fatalf("unexpected effective split: %+v", info)
	}
	if info.TQBlockSize != 0 || info.TQLayoutKind != turboquant.ReferenceLayoutKind {
		t.Fatalf("unexpected layout metadata: %+v", info)
	}
}

func TestKVCacheRuntimeInfoNativeGroupedScaffold(t *testing.T) {
	info := (&InputCache{
		turboQuantPathKind: "native_grouped_scaffold",
		tqBlockSize:        turboquant.NativeGroupSize,
		tqLayoutKind:       turboquant.NativeLayoutKind128,
		tqLayoutVersion:    turboquant.NativeLayoutVersion,
		tqGroupCount:       2,
		tqOriginalHeadDim:  130,
		tqTailPad:          126,
	}).RuntimeInfo()

	if info.TurboQuantPathKind != "native_grouped_scaffold" {
		t.Fatalf("unexpected path kind: %+v", info)
	}
	if info.TQLayoutKind != turboquant.NativeLayoutKind128 || info.TQBlockSize != turboquant.NativeGroupSize {
		t.Fatalf("unexpected grouped layout metadata: %+v", info)
	}
	if info.TQGroupCount != 2 || info.TQOriginalHeadDim != 130 || info.TQTailPad != 126 {
		t.Fatalf("unexpected grouped dimensions: %+v", info)
	}
}

func TestFindCacheSlot(t *testing.T) {
	type expected struct {
		result int
		len    int32
	}

	tests := []struct {
		name    string
		cache   InputCache
		prompt  []*input.Input
		longest expected
		best    expected
	}{
		{
			name: "Empty",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []*input.Input{},
					InUse:    false,
					lastUsed: time.Time{},
				},
				{
					Id:       1,
					Inputs:   []*input.Input{},
					InUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:  []*input.Input{{Token: 1}},
			longest: expected{result: 0, len: 0},
			best:    expected{result: 0, len: 0},
		},
		{
			name: "Extend",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []*input.Input{{Token: 1}},
					InUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
					InUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:  []*input.Input{{Token: 1}, {Token: 2}},
			longest: expected{result: 1, len: 2},
			best:    expected{result: 1, len: 2},
		},
		{
			name: "New",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
					InUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []*input.Input{},
					InUse:    false,
					lastUsed: time.Time{},
				},
			}},
			prompt:  []*input.Input{{Token: 2}},
			longest: expected{result: 0, len: 0},
			best:    expected{result: 1, len: 0},
		},
		{
			name: "Fork",
			cache: InputCache{
				slots: []InputCacheSlot{
					{
						Id:       0,
						Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
						InUse:    false,
						lastUsed: time.Now().Add(-time.Second),
					},
					{
						Id:       1,
						Inputs:   []*input.Input{},
						InUse:    false,
						lastUsed: time.Time{},
					},
				},
			},
			prompt:  []*input.Input{{Token: 1}},
			longest: expected{result: 0, len: 1},
			best:    expected{result: 1, len: 1},
		},
		{
			name: "Evict",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []*input.Input{{Token: 1}},
					InUse:    false,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
					InUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:  []*input.Input{{Token: 2}, {Token: 3}},
			longest: expected{result: 0, len: 0},
			best:    expected{result: 1, len: 0},
		},
		{
			name: "In use",
			cache: InputCache{slots: []InputCacheSlot{
				{
					Id:       0,
					Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
					InUse:    true,
					lastUsed: time.Now().Add(-time.Second),
				},
				{
					Id:       1,
					Inputs:   []*input.Input{{Token: 1}},
					InUse:    false,
					lastUsed: time.Now().Add(-2 * time.Second),
				},
			}},
			prompt:  []*input.Input{{Token: 1}, {Token: 2}},
			longest: expected{result: 1, len: 1},
			best:    expected{result: 1, len: 2},
		},
	}

	for _, tt := range tests {
		t.Run("Longest-"+tt.name, func(t *testing.T) {
			result, resultLen, err := tt.cache.findLongestCacheSlot(tt.prompt)
			if err != nil {
				t.Errorf("findLongestCacheSlot: err %v", err)
			} else if result.Id != tt.longest.result || resultLen != tt.longest.len {
				t.Errorf("findLongestCacheSlot: slot have %v, want %v len have %v, want %v",
					result.Id, tt.longest.result, resultLen, tt.longest.len)
			}
		})
	}

	for _, tt := range tests {
		t.Run("Best-"+tt.name, func(t *testing.T) {
			result, resultLen, err := tt.cache.findBestCacheSlot(tt.prompt)
			if err != nil {
				t.Errorf("findBestCacheSlot: err %v", err)
			} else if result.Id != tt.best.result || resultLen != tt.best.len {
				t.Errorf("findBestCacheSlot: slot have %v, want %v len have %v, want %v",
					result.Id, tt.best.result, resultLen, tt.best.len)
			}
		})
	}
}

func TestShiftDiscard(t *testing.T) {
	tests := []struct {
		name     string
		numCtx   int32
		numKeep  int32
		inputs   []*input.Input
		expected int32
	}{
		{
			name:     "Shift",
			numCtx:   2048,
			numKeep:  5,
			inputs:   slices.Repeat([]*input.Input{{}}, 2048),
			expected: 1021,
		},
		{
			name:     "Max Keep",
			numCtx:   2048,
			numKeep:  2047,
			inputs:   slices.Repeat([]*input.Input{{}}, 2048),
			expected: 1,
		},
		{
			name:     "No Keep",
			numCtx:   2048,
			numKeep:  0,
			inputs:   slices.Repeat([]*input.Input{{}}, 2048),
			expected: 1024,
		},
		{
			name:     "Truncate",
			numCtx:   2048,
			numKeep:  5,
			inputs:   slices.Repeat([]*input.Input{{}}, 5000),
			expected: 3973,
		},
		{
			name:     "Truncate Keep",
			numCtx:   2048,
			numKeep:  2047,
			inputs:   slices.Repeat([]*input.Input{{}}, 5000),
			expected: 2953,
		},
		{
			name:     "No Op",
			numCtx:   2048,
			numKeep:  5,
			inputs:   slices.Repeat([]*input.Input{{}}, 512),
			expected: 0,
		},
		{
			name:    "Same Batch",
			numCtx:  2048,
			numKeep: 5,
			inputs: slices.Collect(func(yield func(*input.Input) bool) {
				for range 1024 {
					if !yield(&input.Input{}) {
						return
					}
				}

				if !yield(&input.Input{SameBatch: 512 - 1}) {
					return
				}

				for range 2048 - 1024 - 1 {
					if !yield(&input.Input{}) {
						return
					}
				}
			}),
			expected: 1531,
		},
		{
			name:    "Same Batch Near Start",
			numCtx:  2048,
			numKeep: 5,
			inputs: slices.Collect(func(yield func(*input.Input) bool) {
				for range 10 {
					if !yield(&input.Input{}) {
						return
					}
				}

				if !yield(&input.Input{SameBatch: 512 - 1}) {
					return
				}

				for range 2048 - 10 - 1 {
					if !yield(&input.Input{}) {
						return
					}
				}
			}),
			expected: 1021,
		},
		{
			name:   "Consecutive Same Batch",
			numCtx: 32,
			inputs: slices.Collect(func(yield func(*input.Input) bool) {
				for i := range 32 {
					input := input.Input{}
					if i%10 == 0 {
						input.SameBatch = 10 - 1
					}
					if !yield(&input) {
						return
					}
				}
			}),
			expected: 20,
		},
		{
			name:   "Overlapping Same Batch",
			numCtx: 32,
			inputs: slices.Collect(func(yield func(*input.Input) bool) {
				for i := range 32 {
					input := input.Input{}
					if slices.Contains([]int{4, 8, 14}, i) {
						input.SameBatch = 10 - 1
					}
					if !yield(&input) {
						return
					}
				}
			}),
			expected: 24,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := InputCache{numCtx: tt.numCtx}
			result := c.ShiftDiscard(tt.inputs, tt.numKeep)
			if result != tt.expected {
				t.Errorf("shiftDiscard(ctx: %v, keep: %v inputs: %v): have %v; want %v", tt.numCtx, tt.numKeep, len(tt.inputs), result, tt.expected)
			}
		})
	}
}

func TestLoadCacheSlot(t *testing.T) {
	tests := []struct {
		name           string
		cache          InputCache
		prompt         []*input.Input
		wantErr        bool
		expectedSlotId int
		expectedPrompt int // expected length of remaining prompt
	}{
		{
			name: "Basic cache hit - single user",
			cache: InputCache{
				multiUserCache: false,
				slots: []InputCacheSlot{
					{
						Id:       0,
						Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
						InUse:    false,
						lastUsed: time.Now().Add(-time.Second),
					},
					{
						Id:       1,
						Inputs:   []*input.Input{},
						InUse:    false,
						lastUsed: time.Now().Add(-2 * time.Second),
					},
				},
			},
			prompt:         []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}},
			wantErr:        false,
			expectedSlotId: 0,
			expectedPrompt: 1, // Only token 3 remains
		},
		{
			name: "Basic cache hit - multi user",
			cache: InputCache{
				multiUserCache: true,
				slots: []InputCacheSlot{
					{
						Id:       0,
						Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
						InUse:    false,
						lastUsed: time.Now().Add(-time.Second),
					},
					{
						Id:       1,
						Inputs:   []*input.Input{},
						InUse:    false,
						lastUsed: time.Now().Add(-2 * time.Second),
					},
				},
			},
			prompt:         []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}},
			wantErr:        false,
			expectedSlotId: 0,
			expectedPrompt: 1, // Only token 3 remains
		},
		{
			name: "Exact match - leave one input",
			cache: InputCache{
				multiUserCache: false,
				slots: []InputCacheSlot{
					{
						Id:       0,
						Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
						InUse:    false,
						lastUsed: time.Now().Add(-time.Second),
					},
				},
			},
			prompt:         []*input.Input{{Token: 1}, {Token: 2}},
			wantErr:        false,
			expectedSlotId: 0,
			expectedPrompt: 1, // Should leave 1 token for sampling
		},
		{
			name: "No available slots",
			cache: InputCache{
				multiUserCache: false,
				slots: []InputCacheSlot{
					{
						Id:       0,
						Inputs:   []*input.Input{{Token: 1}, {Token: 2}},
						InUse:    true,
						lastUsed: time.Now().Add(-time.Second),
					},
				},
			},
			prompt:         []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}},
			wantErr:        true,
			expectedSlotId: -1,
			expectedPrompt: -1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			slot, remainingPrompt, err := tt.cache.LoadCacheSlot(tt.prompt, true)

			// Check error state
			if (err != nil) != tt.wantErr {
				t.Errorf("LoadCacheSlot() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return // Skip further checks if we expected an error
			}

			// Verify slot ID
			if slot.Id != tt.expectedSlotId {
				t.Errorf("LoadCacheSlot() slot ID = %v, expected %v", slot.Id, tt.expectedSlotId)
			}

			// Verify slot is now marked in use
			if !slot.InUse {
				t.Errorf("LoadCacheSlot() slot not marked InUse")
			}

			// Verify remaining prompt length
			if len(remainingPrompt) != tt.expectedPrompt {
				t.Errorf("LoadCacheSlot() remaining prompt length = %v, expected %v",
					len(remainingPrompt), tt.expectedPrompt)
			}
		})
	}
}

// Mock implementation of the Cache interface
type mockCache struct {
	shouldFail bool
}

// Implement only the methods needed for the test
func (m *mockCache) Remove(seq int, beginIndex, endIndex int32) error {
	if m.shouldFail {
		return fmt.Errorf("mock cache removal error")
	}
	return nil
}

// Stub implementations for other interface methods
func (m *mockCache) SetLayer(layer int)                                                            {}
func (m *mockCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor)                          { return nil, nil, nil }
func (m *mockCache) Put(ctx ml.Context, key, value ml.Tensor)                                      {}
func (m *mockCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {}
func (m *mockCache) Close()                                                                        {}
func (m *mockCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error            { return nil }
func (m *mockCache) CopyPrefix(srcSeq, dstSeq int, len int32)                                      {}
func (m *mockCache) SetConfig(ml.CacheConfig)                                                      {}
func (m *mockCache) CanResume(seq int, pos int32) bool                                             { return true }

func TestShiftCacheSlot(t *testing.T) {
	tests := []struct {
		name          string
		numCtx        int32
		inputs        []*input.Input
		numKeep       int32
		cacheErr      bool
		wantErr       any
		wantInputsLen int
	}{
		{
			name:          "Normal shift",
			numCtx:        10,
			inputs:        []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}, {Token: 4}, {Token: 5}, {Token: 6}, {Token: 7}, {Token: 8}, {Token: 9}, {Token: 10}},
			numKeep:       2,
			cacheErr:      false, // No error
			wantErr:       nil,
			wantInputsLen: 6, // After discarding 4 tokens
		},
		{
			name:          "Cache removal fails",
			numCtx:        10,
			inputs:        []*input.Input{{Token: 1}, {Token: 2}, {Token: 3}, {Token: 4}, {Token: 5}, {Token: 6}, {Token: 7}, {Token: 8}, {Token: 9}, {Token: 10}},
			numKeep:       2,
			cacheErr:      true,
			wantErr:       &ErrReprocessInputs{},
			wantInputsLen: 0, // Original inputs should be cleared
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := &mockCache{shouldFail: tt.cacheErr}
			c := InputCache{
				numCtx: tt.numCtx,
				cache:  mock,
			}
			slot := &InputCacheSlot{
				Id:     123,
				Inputs: make([]*input.Input, len(tt.inputs)),
			}
			copy(slot.Inputs, tt.inputs)

			err := c.ShiftCacheSlot(slot, tt.numKeep)

			if tt.wantErr != nil {
				if err == nil {
					t.Errorf("Expected error but got nil")
					return
				}

				if !errors.As(err, &tt.wantErr) {
					t.Errorf("Expected error of type %T but got %T: %v", tt.wantErr, err, err)
				}
			} else if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if len(slot.Inputs) != tt.wantInputsLen {
				t.Errorf("Slot inputs length after operation: got %v, want %v", len(slot.Inputs), tt.wantInputsLen)
			}
		})
	}
}

type runnerTestBackend struct {
	supportsTurboQuantFastPath bool
	supportsTurboQuantCUDA     bool
}

func (b *runnerTestBackend) Close() {}

func (b *runnerTestBackend) Load(ctx context.Context, progress func(float32)) error {
	return nil
}

func (b *runnerTestBackend) BackendMemory() ml.BackendMemory {
	return ml.BackendMemory{}
}

func (b *runnerTestBackend) Config() fs.Config {
	return runnerTestConfig{}
}

func (b *runnerTestBackend) Get(name string) ml.Tensor {
	return nil
}

func (b *runnerTestBackend) NewContext() ml.Context {
	return nil
}

func (b *runnerTestBackend) NewContextSize(size int) ml.Context {
	return nil
}

func (b *runnerTestBackend) BackendDevices() []ml.DeviceInfo {
	return nil
}

func (b *runnerTestBackend) CacheConfig() ml.CacheConfig {
	return ml.CacheConfig{}
}

func (b *runnerTestBackend) TurboQuantSupport() ml.TurboQuantSupport {
	return ml.TurboQuantSupport{CPU: b.supportsTurboQuantFastPath, CUDA: b.supportsTurboQuantCUDA}
}

type runnerTestConfig struct{}

func (runnerTestConfig) Architecture() string                  { return "test" }
func (runnerTestConfig) String(string, ...string) string       { return "" }
func (runnerTestConfig) Uint(string, ...uint32) uint32         { return 0 }
func (runnerTestConfig) Float(string, ...float32) float32      { return 0 }
func (runnerTestConfig) Bool(string, ...bool) bool             { return false }
func (runnerTestConfig) Strings(string, ...[]string) []string  { return nil }
func (runnerTestConfig) Ints(string, ...[]int32) []int32       { return nil }
func (runnerTestConfig) Floats(string, ...[]float32) []float32 { return nil }
func (runnerTestConfig) Bools(string, ...[]bool) []bool        { return nil }
func (runnerTestConfig) Len() int                              { return 0 }
func (runnerTestConfig) Keys() iter.Seq[string]                { return func(yield func(string) bool) {} }
func (runnerTestConfig) Value(string) any                      { return nil }

type runnerTestModel struct {
	model.Base
}

func (m *runnerTestModel) Forward(ctx ml.Context, batch input.Batch) (ml.Tensor, error) {
	return nil, nil
}

func newRunnerTestModel(cache kvcache.Cache, backend ml.Backend) model.Model {
	m := &runnerTestModel{}
	baseValue := reflect.ValueOf(&m.Base).Elem()
	setUnexportedField(baseValue.FieldByName("b"), backend)
	configField := baseValue.FieldByName("config")
	setUnexportedField(configField.FieldByName("Cache"), cache)
	return m
}

func setUnexportedField(field reflect.Value, value any) {
	reflect.NewAt(field.Type(), unsafe.Pointer(field.UnsafeAddr())).Elem().Set(reflect.ValueOf(value))
}

func TestNewInputCacheWrapsTurboQuantCausalCaches(t *testing.T) {
	backend := &runnerTestBackend{supportsTurboQuantFastPath: true}
	model := newRunnerTestModel(kvcache.NewCausalCache(nil), backend)

	inputCache, err := NewInputCache(model, "tq35", "", 16, 1, 1, false)
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := inputCache.cache.(*kvcache.TurboQuantCache); !ok {
		t.Fatalf("cache type = %T, want *kvcache.TurboQuantCache", inputCache.cache)
	}
}

func TestNewInputCachePreservesWrapperNonCausalCaches(t *testing.T) {
	backend := &runnerTestBackend{supportsTurboQuantFastPath: true}
	model := newRunnerTestModel(
		kvcache.NewWrapperCache(
			kvcache.NewEncoderCache(),
			kvcache.NewCausalCache(nil),
		),
		backend,
	)

	inputCache, err := NewInputCache(model, "tq25", "", 16, 1, 1, false)
	if err != nil {
		t.Fatal(err)
	}

	wrapper, ok := inputCache.cache.(*kvcache.WrapperCache)
	if !ok {
		t.Fatalf("cache type = %T, want *kvcache.WrapperCache", inputCache.cache)
	}
	if _, ok := wrapper.UnderlyingCache().(*kvcache.EncoderCache); !ok {
		t.Fatalf("wrapper cache[0] type = %T, want *kvcache.EncoderCache", wrapper.UnderlyingCache())
	}
	wrapper.SetLayerType(1)
	if _, ok := wrapper.UnderlyingCache().(*kvcache.TurboQuantCache); !ok {
		t.Fatalf("wrapper cache[1] type = %T, want *kvcache.TurboQuantCache", wrapper.UnderlyingCache())
	}
}

func TestNewInputCacheLeavesNonTurboQuantModesUnwrapped(t *testing.T) {
	backend := &runnerTestBackend{}
	model := newRunnerTestModel(kvcache.NewCausalCache(nil), backend)

	inputCache, err := NewInputCache(model, "q4_0", "", 16, 1, 1, false)
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := inputCache.cache.(*kvcache.TurboQuantCache); ok {
		t.Fatalf("cache type = %T, did not expect TurboQuant wrapping", inputCache.cache)
	}
	if _, ok := inputCache.cache.(*kvcache.Causal); !ok {
		t.Fatalf("cache type = %T, want *kvcache.Causal", inputCache.cache)
	}
}

func TestNewInputCacheFallsBackToDenseWhenTurboQuantFastPathUnsupported(t *testing.T) {
	backend := &runnerTestBackend{supportsTurboQuantFastPath: false}
	model := newRunnerTestModel(kvcache.NewCausalCache(nil), backend)

	inputCache, err := NewInputCache(model, "tq35", "", 16, 1, 1, false)
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := inputCache.cache.(*kvcache.TurboQuantCache); !ok {
		t.Fatalf("cache type = %T, want *kvcache.TurboQuantCache", inputCache.cache)
	}
	info := inputCache.RuntimeInfo()
	if info.Algorithm != "paper" {
		t.Fatalf("runtime algorithm = %q, want paper", info.Algorithm)
	}
	if info.Path != "dense-fallback" {
		t.Fatalf("runtime path = %q, want dense-fallback", info.Path)
	}
}

func TestNewInputCacheTracksExplicitCUDARequestWithoutCUDAFastPath(t *testing.T) {
	backend := &runnerTestBackend{supportsTurboQuantFastPath: true}
	model := newRunnerTestModel(kvcache.NewCausalCache(nil), backend)

	inputCache, err := NewInputCache(model, "tq35", "cuda", 16, 1, 1, false)
	if err != nil {
		t.Fatal(err)
	}

	info := inputCache.RuntimeInfo()
	if info.Algorithm != "paper" {
		t.Fatalf("runtime algorithm = %q, want paper", info.Algorithm)
	}
	if info.Backend != "cuda" {
		t.Fatalf("runtime backend = %q, want cuda", info.Backend)
	}
	if info.Path != "dense-fallback" {
		t.Fatalf("runtime path = %q, want dense-fallback", info.Path)
	}
}
