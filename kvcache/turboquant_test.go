package kvcache

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

type fakePackedKVHandle struct {
	pathKind         string
	ownsPackedK      bool
	ownsPackedV      bool
	supportsFastGetK bool
	supportsFastGetV bool
}

func (h fakePackedKVHandle) PathKind() string       { return h.pathKind }
func (h fakePackedKVHandle) OwnsPackedK() bool      { return h.ownsPackedK }
func (h fakePackedKVHandle) OwnsPackedV() bool      { return h.ownsPackedV }
func (h fakePackedKVHandle) SupportsFastGetK() bool { return h.supportsFastGetK }
func (h fakePackedKVHandle) SupportsFastGetV() bool { return h.supportsFastGetV }
func (h fakePackedKVHandle) Close() error           { return nil }

type decodedSequenceEntry struct {
	pos   int32
	key   []float32
	value []float32
}

func TestTurboQuantCacheStoreRoundTrip(t *testing.T) {
	presets := []turboquant.Preset{turboquant.PresetTQ25, turboquant.PresetTQ35}
	for _, preset := range presets {
		t.Run(preset.Name, func(t *testing.T) {
			runPermutedVariants(t, func(t *testing.T, backend *testBackend) {
				// Exercise both PermutedV layouts and a keyDim != valueDim shape so
				// value encoding does not accidentally reuse the key stride.
				cache := NewTurboQuantCache(NewCausalCache(nil), preset, "")
				defer cache.Close()

				dtype := ml.DTypeTQ35
				if preset.Name == "tq25" {
					dtype = ml.DTypeTQ25
				}
				cache.Init(backend, dtype, 1, 16, 16)

				ctx := backend.NewContext()
				defer ctx.Close()

				mustStartForward(t, cache, ctx, []int32{0, 1}, []int{0, 0})
				cache.SetLayer(0)

				keyValues := []float32{
					1, 11, 2, 12,
					3, 13, 4, 14,
				}
				valueValues := []float32{
					21, 31, 22, 32,
					23, 33, 24, 34,
					25, 35, 26, 36,
				}

				keyTensor := ctx.FromFloats(keyValues, 2, 2, 2)
				valueTensor := ctx.FromFloats(valueValues, 3, 2, 2)
				cache.Put(ctx, keyTensor, valueTensor)

				key, value, mask := cache.Get(ctx)
				if !slices.Equal(key.Shape(), []int{2, 2, 2}) {
					t.Fatalf("key shape = %v, want [2 2 2]", key.Shape())
				}
				maxValueMSE := float32(50)
				if preset.Name == "tq25" {
					maxValueMSE = 90
				}
				if backend.permutedV {
					if !slices.Equal(value.Shape(), []int{2, 3, 2}) {
						t.Fatalf("value shape = %v, want [2 3 2]", value.Shape())
					}
					if mse(permuteValueRows(valueValues, 3, 2, 2), value.Floats()) > maxValueMSE {
						t.Fatalf("permuted value mse = %v, want <= %v", mse(permuteValueRows(valueValues, 3, 2, 2), value.Floats()), maxValueMSE)
					}
				} else {
					if !slices.Equal(value.Shape(), []int{3, 2, 2}) {
						t.Fatalf("value shape = %v, want [3 2 2]", value.Shape())
					}
					if mse(valueValues, value.Floats()) > maxValueMSE {
						t.Fatalf("value mse = %v, want <= %v", mse(valueValues, value.Floats()), maxValueMSE)
					}
				}

				if mse(keyValues, key.Floats()) > 50 {
					t.Fatalf("key mse = %v, want <= 50", mse(keyValues, key.Floats()))
				}

				wantMask := []float32{
					0, float32(math.Inf(-1)),
					0, 0,
				}
				if !slices.Equal(mask.Floats(), wantMask) {
					t.Fatalf("mask = %v, want %v", mask.Floats(), wantMask)
				}
			})
		})
	}
}

func TestTurboQuantCacheStoresPaperFormatRows(t *testing.T) {
	presets := []turboquant.Preset{turboquant.PresetTQ25, turboquant.PresetTQ35}
	for _, preset := range presets {
		t.Run(preset.Name, func(t *testing.T) {
			cache := NewTurboQuantCache(NewCausalCache(nil), preset, "")
			defer cache.Close()

			dtype := ml.DTypeTQ35
			if preset.Name == "tq25" {
				dtype = ml.DTypeTQ25
			}

			backend := &testBackend{}
			cache.Init(backend, dtype, 1, 16, 16)

			ctx := backend.NewContext()
			defer ctx.Close()

			mustStartForward(t, cache, ctx, []int32{0}, []int{0})
			cache.SetLayer(0)
			cache.Put(
				ctx,
				ctx.FromFloats([]float32{1, 2}, 1, 1, 2),
				ctx.FromFloats([]float32{11, 12}, 1, 1, 2),
			)

			entry := cache.data[0][0]
			if len(entry.key.Data) == 0 || len(entry.value.Data) == 0 {
				t.Fatal("expected packed key and value rows")
			}
			if entry.key.LayoutKind != turboquant.ReferenceLayoutKind || entry.value.LayoutKind != turboquant.ReferenceLayoutKind {
				t.Fatalf("expected reference layout rows, got key=%q value=%q", entry.key.LayoutKind, entry.value.LayoutKind)
			}

			keyVector, err := turboquant.UnmarshalEncodedVector(entry.key.Data)
			if err != nil {
				t.Fatalf("unmarshal key vector: %v", err)
			}
			if len(keyVector.Blocks) != 1 {
				t.Fatalf("key vector block count = %d, want 1", len(keyVector.Blocks))
			}
			keyBlock := keyVector.Blocks[0]
			if keyBlock.Version != turboquant.BlockVersion {
				t.Fatalf("key block version = %d, want %d", keyBlock.Version, turboquant.BlockVersion)
			}
			if keyBlock.PresetID != preset.ID {
				t.Fatalf("key preset id = %d, want %d", keyBlock.PresetID, preset.ID)
			}
			if keyBlock.RegularBits != uint8(preset.KeyPrimaryBits) {
				t.Fatalf("key bits = %d, want %d", keyBlock.RegularBits, preset.KeyPrimaryBits)
			}
			if keyBlock.QJLRows == 0 {
				t.Fatal("expected product-mode key block to include QJL rows")
			}

			valueVector, err := turboquant.UnmarshalEncodedVector(entry.value.Data)
			if err != nil {
				t.Fatalf("unmarshal value vector: %v", err)
			}
			if len(valueVector.Blocks) != 1 {
				t.Fatalf("value vector block count = %d, want 1", len(valueVector.Blocks))
			}
			valueBlock := valueVector.Blocks[0]
			if valueBlock.Version != turboquant.BlockVersion {
				t.Fatalf("value block version = %d, want %d", valueBlock.Version, turboquant.BlockVersion)
			}
			if valueBlock.PresetID != preset.ID {
				t.Fatalf("value preset id = %d, want %d", valueBlock.PresetID, preset.ID)
			}
			if valueBlock.RegularBits != uint8(preset.ValueBits) {
				t.Fatalf("value bits = %d, want %d", valueBlock.RegularBits, preset.ValueBits)
			}
			if valueBlock.QJLRows != 0 {
				t.Fatalf("value QJL rows = %d, want 0", valueBlock.QJLRows)
			}
		})
	}
}

func TestTurboQuantCacheStoresNativeGroupedRowsWhenExplicitlyEnabled(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	cache.storageLayoutKind = turboquant.NativeLayoutKind128
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0}, []int{0})
	cache.SetLayer(0)
	cache.Put(
		ctx,
		ctx.FromFloats(make([]float32, 130), 130, 1, 1),
		ctx.FromFloats(make([]float32, 130), 130, 1, 1),
	)

	entry := cache.data[0][0]
	if entry.key.LayoutKind != turboquant.NativeLayoutKind128 || entry.value.LayoutKind != turboquant.NativeLayoutKind128 {
		t.Fatalf("expected grouped native layout rows, got key=%q value=%q", entry.key.LayoutKind, entry.value.LayoutKind)
	}

	var keyPayload turboquant.NativeGroupedVector
	if err := keyPayload.UnmarshalBinary(entry.key.Data); err != nil {
		t.Fatalf("unmarshal grouped key payload: %v", err)
	}
	if keyPayload.Header.GroupCount != 2 || keyPayload.Header.OriginalHeadDim != 130 {
		t.Fatalf("unexpected key grouped header: %+v", keyPayload.Header)
	}
	if cache.layoutInfo.LayoutKind != turboquant.NativeLayoutKind128 || cache.layoutInfo.BlockSize != turboquant.NativeGroupSize {
		t.Fatalf("unexpected cache layout info: %+v", cache.layoutInfo)
	}
}

func TestTurboQuantCacheBackendStatusDefaultsToReferenceWrapper(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	cache.Init(&testBackend{}, ml.DTypeTQ35, 1, 16, 16)

	status := cache.TurboQuantBackendStatus()
	if status.PathKind != "reference_wrapper" {
		t.Fatalf("path kind = %q, want reference_wrapper", status.PathKind)
	}
	if status.BackendPackedKOwned || status.BackendPackedVOwned {
		t.Fatalf("unexpected backend ownership status: %+v", status)
	}
	if status.NativeBackendReady {
		t.Fatalf("native backend ready = true, want false")
	}
	if status.NativeBackendBlocker == "" {
		t.Fatal("expected explicit blocker for backend-native ownership")
	}
}

func TestTurboQuantCacheBackendStatusReflectsAttachedHandle(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	cache.Init(&testBackend{}, ml.DTypeTQ35, 1, 16, 16)
	cache.backendPackedHandle = fakePackedKVHandle{
		pathKind:         "native_backend",
		ownsPackedK:      true,
		ownsPackedV:      false,
		supportsFastGetK: true,
	}
	cache.backendPackedReady = true
	cache.backendPackedKOwned = true
	cache.backendPackedVOwned = false
	cache.backendPackedKReady = true
	cache.backendPackedVReady = false
	cache.backendPackedBlocker = "backend-native packed V ownership is still scaffolded"

	status := cache.TurboQuantBackendStatus()
	if status.PathKind != "native_backend" {
		t.Fatalf("path kind = %q, want native_backend", status.PathKind)
	}
	if !status.BackendPackedKOwned || status.BackendPackedVOwned {
		t.Fatalf("unexpected backend ownership status: %+v", status)
	}
	if !status.NativeBackendReady {
		t.Fatalf("native backend ready = false, want true")
	}
	if status.NativeBackendBlocker == "" {
		t.Fatal("expected explicit V-side blocker")
	}
}

func TestTurboQuantCacheMultiBatchAppend(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1}, []int{0, 0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{1, 2}, 1, 1, 2), ctx.FromFloats([]float32{11, 12}, 1, 1, 2))

	mustStartForward(t, cache, ctx, []int32{2}, []int{0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{3}, 1, 1, 1), ctx.FromFloats([]float32{13}, 1, 1, 1))

	key, value, mask := cache.Get(ctx)
	if !slices.Equal(key.Shape(), []int{1, 1, 3}) {
		t.Fatalf("key shape = %v, want [1 1 3]", key.Shape())
	}
	if !slices.Equal(value.Shape(), []int{1, 1, 3}) {
		t.Fatalf("value shape = %v, want [1 1 3]", value.Shape())
	}
	if mse([]float32{1, 2, 3}, key.Floats()) > 25 {
		t.Fatalf("key mse = %v, want <= 25", mse([]float32{1, 2, 3}, key.Floats()))
	}
	if mse([]float32{11, 12, 13}, value.Floats()) > 25 {
		t.Fatalf("value mse = %v, want <= 25", mse([]float32{11, 12, 13}, value.Floats()))
	}
	if !slices.Equal(mask.Floats(), []float32{0, 0, 0}) {
		t.Fatalf("mask = %v, want [0 0 0]", mask.Floats())
	}
}

func TestTurboQuantCacheCopyPrefixAndResume(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ25, "")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ25, 2, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1}, []int{0, 0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{1, 2}, 1, 1, 2), ctx.FromFloats([]float32{11, 12}, 1, 1, 2))

	if got := countStoredEntries(cache, 0); got != 2 {
		t.Fatalf("stored entries before copy = %d, want 2", got)
	}

	cache.CopyPrefix(0, 1, 2)

	if got := countStoredEntries(cache, 0); got != 2 {
		t.Fatalf("stored entries after copy = %d, want 2", got)
	}
	if !cache.CanResume(1, 2) {
		t.Fatal("expected copied prefix to be resumable")
	}

	srcEntries := decodeSequenceEntries(t, cache, 0, 0)
	dstEntries := decodeSequenceEntries(t, cache, 0, 1)
	if len(srcEntries) != 2 || len(dstEntries) != 2 {
		t.Fatalf("decoded entries = (%d, %d), want (2, 2)", len(srcEntries), len(dstEntries))
	}
	for i := range srcEntries {
		if srcEntries[i].pos != dstEntries[i].pos {
			t.Fatalf("entry %d pos mismatch: got %d want %d", i, dstEntries[i].pos, srcEntries[i].pos)
		}
		if mse(srcEntries[i].key, dstEntries[i].key) > 1e-6 {
			t.Fatalf("entry %d key mse = %v, want 0", i, mse(srcEntries[i].key, dstEntries[i].key))
		}
	}
}

func TestTurboQuantCacheRemoveTailClearsPackedEntries(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1, 2}, []int{0, 0, 0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{1, 2, 3}, 1, 1, 3), ctx.FromFloats([]float32{11, 12, 13}, 1, 1, 3))

	if err := cache.Remove(0, 2, math.MaxInt32); err != nil {
		t.Fatal(err)
	}

	entries := decodeSequenceEntries(t, cache, 0, 0)
	if len(entries) != 2 {
		t.Fatalf("remaining entries = %d, want 2", len(entries))
	}
	if entries[0].pos != 0 || entries[1].pos != 1 {
		t.Fatalf("remaining positions = [%d %d], want [0 1]", entries[0].pos, entries[1].pos)
	}
	if got := countStoredEntries(cache, 0); got != 2 {
		t.Fatalf("stored entries = %d, want 2", got)
	}
}

func TestTurboQuantCacheRemoveMiddleShiftsKeys(t *testing.T) {
	shiftFn := func(ctx ml.Context, layer int, key, shift ml.Tensor) (ml.Tensor, error) {
		values := key.Floats()
		out := make([]float32, len(values))
		offset := shift.Floats()[0]
		for i, v := range values {
			out[i] = v + offset
		}
		return ctx.Input().FromFloats(out, key.Shape()...), nil
	}

	cache := NewTurboQuantCache(NewCausalCache(shiftFn), turboquant.PresetTQ35, "")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1, 2}, []int{0, 0, 0})
	cache.SetLayer(0)
	cache.Put(
		ctx,
		ctx.FromFloats([]float32{10, 20, 30, 11, 21, 31}, 2, 1, 3),
		ctx.FromFloats([]float32{100, 200, 300}, 1, 1, 3),
	)

	if err := cache.Remove(0, 1, 2); err != nil {
		t.Fatal(err)
	}

	entries := decodeSequenceEntries(t, cache, 0, 0)
	if len(entries) != 2 {
		t.Fatalf("remaining entries = %d, want 2", len(entries))
	}
	if entries[0].pos != 0 || entries[1].pos != 1 {
		t.Fatalf("remaining positions = [%d %d], want [0 1]", entries[0].pos, entries[1].pos)
	}
	if mse([]float32{10, 11}, entries[0].key) > 80 {
		t.Fatalf("first key mse = %v, want <= 80", mse([]float32{10, 11}, entries[0].key))
	}
	if mse([]float32{29, 30}, entries[1].key) > 600 {
		t.Fatalf("shifted key mse = %v, want <= 600", mse([]float32{29, 30}, entries[1].key))
	}
	if mse([]float32{300}, entries[1].value) > 3000 {
		t.Fatalf("shifted value mse = %v, want <= 3000", mse([]float32{300}, entries[1].value))
	}
}

func TestTurboQuantCacheRemoveMiddleRequiresShiftSupport(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1, 2}, []int{0, 0, 0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{1, 2, 3}, 1, 1, 3), ctx.FromFloats([]float32{11, 12, 13}, 1, 1, 3))

	if err := cache.Remove(0, 1, 2); err != ErrNotSupported {
		t.Fatalf("Remove returned %v, want %v", err, ErrNotSupported)
	}
}

func TestTurboQuantCacheSWAWindowBehavior(t *testing.T) {
	cache := NewTurboQuantCache(NewSWACache(1, nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1, 2, 3}, []int{0, 0, 0, 0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{1, 2, 3, 4}, 1, 1, 4), ctx.FromFloats([]float32{1, 2, 3, 4}, 1, 1, 4))

	mustStartForward(t, cache, ctx, []int32{4, 5}, []int{0, 0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{5, 6}, 1, 1, 2), ctx.FromFloats([]float32{5, 6}, 1, 1, 2))

	key, _, mask := cache.Get(ctx)
	if !slices.Equal(key.Shape(), []int{1, 1, 4}) {
		t.Fatalf("key shape = %v, want [1 1 4]", key.Shape())
	}
	if mse([]float32{5, 6, 3, 4}, key.Floats()) > 25 {
		t.Fatalf("swa key mse = %v, want <= 25", mse([]float32{5, 6, 3, 4}, key.Floats()))
	}

	wantMask := []float32{
		0, float32(math.Inf(-1)), float32(math.Inf(-1)), 0,
		0, 0, float32(math.Inf(-1)), float32(math.Inf(-1)),
	}
	if !slices.Equal(mask.Floats(), wantMask) {
		t.Fatalf("mask = %v, want %v", mask.Floats(), wantMask)
	}
}

func TestTurboQuantCacheSWAMemResumeBehavior(t *testing.T) {
	cache := NewTurboQuantCache(NewSWAMemCache(4, 5, nil), turboquant.PresetTQ25, "")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ25, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1, 2, 3, 4, 5, 6}, []int{0, 0, 0, 0, 0, 0, 0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{1, 2, 3, 4, 5, 6, 7}, 1, 1, 7), ctx.FromFloats([]float32{1, 2, 3, 4, 5, 6, 7}, 1, 1, 7))

	mustStartForward(t, cache, ctx, []int32{7}, []int{0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{8}, 1, 1, 1), ctx.FromFloats([]float32{8}, 1, 1, 1))

	if cache.CanResume(0, 0) {
		t.Fatal("CanResume(0, 0) = true, want false")
	}
	if !cache.CanResume(0, 6) {
		t.Fatal("CanResume(0, 6) = false, want true")
	}
	if !cache.CanResume(0, 7) {
		t.Fatal("CanResume(0, 7) = false, want true")
	}
}

func TestWrapWithTurboQuantPreservesNonCausalCaches(t *testing.T) {
	wrapper := NewWrapperCache(
		NewEncoderCache(),
		NewCausalCache(nil),
		NewSWACache(4, nil),
	)

	wrapped := WrapWithTurboQuant(wrapper, turboquant.PresetTQ35, "").(*WrapperCache)
	if _, ok := wrapped.caches[0].(*EncoderCache); !ok {
		t.Fatalf("cache[0] type = %T, want *EncoderCache", wrapped.caches[0])
	}
	if _, ok := wrapped.caches[1].(*TurboQuantCache); !ok {
		t.Fatalf("cache[1] type = %T, want *TurboQuantCache", wrapped.caches[1])
	}
	if _, ok := wrapped.caches[2].(*TurboQuantCache); !ok {
		t.Fatalf("cache[2] type = %T, want *TurboQuantCache", wrapped.caches[2])
	}
}

func TestTurboQuantCacheCloseClearsSideStorage(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0}, []int{0})
	cache.SetLayer(0)
	cache.Put(ctx, ctx.FromFloats([]float32{1}, 1, 1, 1), ctx.FromFloats([]float32{11}, 1, 1, 1))

	cache.Close()
	if len(cache.data) != 0 {
		t.Fatalf("len(cache.data) = %d, want 0", len(cache.data))
	}
	if len(cache.shape) != 0 {
		t.Fatalf("len(cache.shape) = %d, want 0", len(cache.shape))
	}
}

func TestTurboQuantCacheGetUsesCompressedKeyFastPath(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	backend := &testBackend{fastPath: true}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1}, []int{0, 0})
	cache.SetLayer(0)
	cache.Put(
		ctx,
		ctx.FromFloats([]float32{1, 11, 2, 12}, 2, 1, 2),
		ctx.FromFloats([]float32{21, 31, 22, 32}, 2, 1, 2),
	)

	key, value, mask := cache.Get(ctx)
	if key == nil || value == nil || mask == nil {
		t.Fatal("expected key, value, and mask tensors")
	}
	if key.DType() != ml.DTypeTQ35 {
		t.Fatalf("key dtype = %v, want %v", key.DType(), ml.DTypeTQ35)
	}
	if !slices.Equal(key.Shape(), []int{len(cache.data[0][0].key.Data), 2}) {
		t.Fatalf("key shape = %v, want [%d 2]", key.Shape(), len(cache.data[0][0].key.Data))
	}
	if !slices.Equal(value.Shape(), []int{2, 1, 2}) {
		t.Fatalf("value shape = %v, want [2 1 2]", value.Shape())
	}
	if len(key.Bytes()) != len(cache.data[0][0].key.Data)*2 {
		t.Fatalf("compressed key byte length = %d, want %d", len(key.Bytes()), len(cache.data[0][0].key.Data)*2)
	}
}

func TestTurboQuantCacheFastPathFallsBackForInconsistentRows(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "")
	defer cache.Close()

	backend := &testBackend{fastPath: true}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	ctx := backend.NewContext()
	defer ctx.Close()

	mustStartForward(t, cache, ctx, []int32{0, 1}, []int{0, 0})
	cache.SetLayer(0)
	cache.Put(
		ctx,
		ctx.FromFloats([]float32{1, 2}, 1, 1, 2),
		ctx.FromFloats([]float32{11, 12}, 1, 1, 2),
	)

	grouped, err := turboquant.EncodeNativeGroupedVector([]float32{2}, turboquant.PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	groupedData, err := grouped.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	cache.data[0][1].key = payloadRow{
		LayoutKind:    turboquant.NativeLayoutKind128,
		LayoutVersion: grouped.Header.LayoutVersion,
		Data:          groupedData,
		GroupCount:    grouped.Header.GroupCount,
		OriginalHead:  grouped.Header.OriginalHeadDim,
		TailPad:       grouped.Header.TailPad,
	}

	key, value, _ := cache.Get(ctx)
	if key.DType() != ml.DTypeF32 {
		t.Fatalf("fallback key dtype = %v, want %v", key.DType(), ml.DTypeF32)
	}
	if !slices.Equal(key.Shape(), []int{1, 1, 2}) {
		t.Fatalf("fallback key shape = %v, want [1 1 2]", key.Shape())
	}
	if !slices.Equal(value.Shape(), []int{1, 1, 2}) {
		t.Fatalf("fallback value shape = %v, want [1 1 2]", value.Shape())
	}
}

func mustStartForward(t *testing.T, cache Cache, ctx ml.Context, positions []int32, sequences []int) {
	t.Helper()
	if err := cache.StartForward(ctx, input.Batch{Positions: positions, Sequences: sequences}, false); err != nil {
		t.Fatal(err)
	}
}

func countStoredEntries(cache *TurboQuantCache, layer int) int {
	count := 0
	for _, entry := range cache.data[layer] {
		if len(entry.key.Data) > 0 || len(entry.value.Data) > 0 {
			count++
		}
	}
	return count
}

func decodeSequenceEntries(t *testing.T, cache *TurboQuantCache, layer, seq int) []decodedSequenceEntry {
	t.Helper()

	var entries []decodedSequenceEntry
	for i, cell := range cache.meta.cells {
		if !slices.Contains(cell.sequences, seq) {
			continue
		}
		entry := cache.data[layer][i]
		if len(entry.key.Data) == 0 || len(entry.value.Data) == 0 {
			continue
		}
		key, err := decodePayloadRow(entry.key)
		if err != nil {
			t.Fatal(err)
		}
		value, err := decodePayloadRow(entry.value)
		if err != nil {
			t.Fatal(err)
		}
		entries = append(entries, decodedSequenceEntry{
			pos:   cell.pos,
			key:   key,
			value: value,
		})
	}

	slices.SortFunc(entries, func(a, b decodedSequenceEntry) int {
		switch {
		case a.pos < b.pos:
			return -1
		case a.pos > b.pos:
			return 1
		default:
			return 0
		}
	})

	return entries
}

func mse(a, b []float32) float32 {
	var out float32
	for i := range a {
		d := a[i] - b[i]
		out += d * d
	}
	return out / float32(len(a))
}

func TestTurboQuantCacheUsesDenseStorageDTypeUnderneath(t *testing.T) {
	cache := NewTurboQuantCache(NewCausalCache(nil), turboquant.PresetTQ35, "cuda")
	defer cache.Close()

	backend := &testBackend{}
	cache.Init(backend, ml.DTypeTQ35, 1, 16, 16)

	if cache.requestedDType != ml.DTypeTQ35 {
		t.Fatalf("requestedDType = %v, want %v", cache.requestedDType, ml.DTypeTQ35)
	}
	if cache.meta.DType != ml.DTypeF16 {
		t.Fatalf("meta.DType = %v, want %v", cache.meta.DType, ml.DTypeF16)
	}
	if cache.requestedBackend != "cuda" {
		t.Fatalf("requestedBackend = %q, want cuda", cache.requestedBackend)
	}
}
