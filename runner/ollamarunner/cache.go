package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/turboquant"
)

type InputCache struct {
	// context window size (per slot)
	numCtx int32

	// does the cache store data or do we need to always send the full input?
	// note that when enabled is false the underlying cache may either be nil
	// or a non-nil dummy that doesn't actually store anything
	enabled bool

	// individual KV caches
	slots []InputCacheSlot

	// optimize cache eviction for multiple users
	multiUserCache bool

	cache kvcache.Cache

	kvCacheRequested          string
	kvCacheEffective          string
	kvCacheRequestedK         string
	kvCacheRequestedV         string
	kvCacheEffectiveK         string
	kvCacheEffectiveV         string
	kvAlgoResolved            string
	kvAlgoResolvedK           string
	kvAlgoResolvedV           string
	kvCacheBackend            string
	kvCachePath               string
	kvCachePathK              string
	kvCachePathV              string
	kvSymmetric               bool
	kvAsymmetric              bool
	fallbackReason            string
	turboQuantPathKind        string
	nativeTurboQuantActive    bool
	referenceTurboQuantActive bool
	backendPackedKOwned       bool
	backendPackedVOwned       bool
	backendPackedKAvailable   bool
	backendPackedVAvailable   bool
	nativeBackendReady        bool
	nativeBackendBlocker      string
	vTurboSupported           bool
	tqBlockSize               int
	tqLayoutKind              string
	tqLayoutVersion           int
	tqGroupCount              int
	tqOriginalHeadDim         int
	tqTailPad                 int
}

func NewInputCache(model model.Model, kvCacheType, kvCacheTypeK, kvCacheTypeV, kvCacheBackend string, kvSize int32, numSlots int, batchSize int, multiUserCache bool) (*InputCache, error) {
	numCtx := kvSize / int32(numSlots)

	if int(numCtx) < batchSize {
		return nil, fmt.Errorf("kv size must be at least as large as batch size * parallel (kv: %v batch: %v parallel: %v)", kvSize, batchSize, numSlots)
	}

	slots := make([]InputCacheSlot, numSlots)

	for i := range slots {
		slots[i] = InputCacheSlot{Id: i}
	}

	cache := model.Config().Cache
	normalizedKVCacheType := normalizeKVCacheType(kvCacheType)
	normalizedKVCacheTypeK, normalizedKVCacheTypeV := resolveKVCacheTypes(kvCacheType, kvCacheTypeK, kvCacheTypeV)
	requestedKVCacheTypeK := normalizedKVCacheTypeK
	requestedKVCacheTypeV := normalizedKVCacheTypeV
	normalizedKVCacheBackend := normalizeKVCacheBackend(kvCacheBackend)
	kvCachePath := "dense-fallback"
	kvCacheEffective := normalizedKVCacheType
	kvCacheEffectiveK := normalizedKVCacheTypeK
	kvCacheEffectiveV := normalizedKVCacheTypeV
	kvAlgoResolved := ""
	kvAlgoResolvedK := ""
	kvAlgoResolvedV := ""
	kvCachePathK := "dense-fallback"
	kvCachePathV := "dense-fallback"
	fallbackReason := ""
	turboQuantPathKind := "disabled"
	nativeTurboQuantActive := false
	referenceTurboQuantActive := false
	backendPackedKOwned := false
	backendPackedVOwned := false
	backendPackedKAvailable := false
	backendPackedVAvailable := false
	nativeBackendReady := false
	nativeBackendBlocker := ""
	vTurboSupported := normalizedKVCacheTypeV == "" || normalizedKVCacheTypeV == "f16"
	tqBlockSize := 0
	tqLayoutKind := ""
	tqLayoutVersion := 0
	tqGroupCount := 0
	tqOriginalHeadDim := 0
	tqTailPad := 0
	if normalizedKVCacheTypeK != normalizedKVCacheTypeV {
		fallbackReason = "asymmetric K/V cache modes requested but native backend ownership is not enabled in this branch; falling back to f16/f16"
		kvCacheEffective = "f16"
		kvCacheEffectiveK = "f16"
		kvCacheEffectiveV = "f16"
		normalizedKVCacheType = "f16"
		normalizedKVCacheTypeK = "f16"
		normalizedKVCacheTypeV = "f16"
	}
	if cache != nil {
		dtype := kvCacheTypeFromStr(normalizedKVCacheType)
		dtypeK := kvCacheTypeFromStr(normalizedKVCacheTypeK)
		dtypeV := kvCacheTypeFromStr(normalizedKVCacheTypeV)
		kvCachePathK = resolveKVCachePath(model.Backend(), dtypeK, normalizedKVCacheBackend)
		kvCachePathV = resolveKVCachePath(model.Backend(), dtypeV, normalizedKVCacheBackend)
		// Implemented explicit asymmetric K/V resolution at cache setup instead of treating everything as symmetric; idea source: @TheTom.
		// Preserved visible mixed K/V request handling so conservative asymmetric pairings remain measurable; idea source: @primoco.
		// Added guarded fallback reporting for risky asymmetric pairings instead of silent degradation; idea source: @sjoerdmaessen.
		if isTurboQuantKVType(normalizedKVCacheTypeV) && kvCachePathV == "dense-fallback" {
			fallbackReason = "requested V turboquant path is not supported by the active backend; falling back to f16 on V"
			normalizedKVCacheTypeV = "f16"
			kvCacheEffectiveV = "f16"
			dtypeV = ml.DTypeF16
			kvCachePathV = "dense-fallback"
			vTurboSupported = false
		}
		if isTurboQuantKVType(normalizedKVCacheTypeK) && kvCachePathK != "dense-fallback" {
			kvAlgoResolvedK = turboquant.AlgorithmPaper
		}
		if isTurboQuantKVType(normalizedKVCacheTypeV) && kvCachePathV != "dense-fallback" {
			kvAlgoResolvedV = turboquant.AlgorithmPaper
			vTurboSupported = true
		}
		if preset, ok := kvcachePreset(dtype); ok {
			cache = kvcache.WrapWithTurboQuant(cache, preset, normalizedKVCacheBackend)
			kvCachePath = resolveKVCachePath(model.Backend(), dtype, normalizedKVCacheBackend)
			kvAlgoResolved = turboquant.AlgorithmPaper
			turboQuantPathKind = "reference_wrapper"
			referenceTurboQuantActive = true
			slog.Info("using turboquant kv cache", "requested", kvCacheType, "backend", normalizedKVCacheBackend, "preset", preset.Name, "path", kvCachePath)
			if kvCachePath == "dense-fallback" {
				slog.Warn("turboquant kv cache requested but backend cannot use requested fast path; falling back to dense attention path",
					"requested", normalizedKVCacheType, "backend", normalizedKVCacheBackend, "path", kvCachePath)
			}
		}
		cache.Init(model.Backend(), dtype, numSlots, int(numCtx), batchSize)
		if info, ok := kvcache.LookupTurboQuantLayoutInfo(cache); ok {
			if info.PathKind != "" {
				turboQuantPathKind = info.PathKind
			}
			if turboQuantPathKind == "native_grouped_scaffold" {
				referenceTurboQuantActive = false
			}
			tqLayoutKind = info.LayoutKind
			tqLayoutVersion = info.LayoutVersion
			tqGroupCount = info.GroupCount
			tqOriginalHeadDim = info.OriginalHeadDim
			tqTailPad = info.TailPad
			tqBlockSize = info.BlockSize
		}
		if status, ok := kvcache.LookupTurboQuantBackendStatus(cache); ok {
			backendPackedKOwned = status.BackendPackedKOwned
			backendPackedVOwned = status.BackendPackedVOwned
			backendPackedKAvailable = status.BackendPackedKReady
			backendPackedVAvailable = status.BackendPackedVReady
			nativeBackendReady = status.NativeBackendReady
			nativeBackendBlocker = status.NativeBackendBlocker
			nativeTurboQuantActive = status.BackendPackedKOwned || status.BackendPackedVOwned
			if nativeTurboQuantActive {
				turboQuantPathKind = "native_backend"
				referenceTurboQuantActive = false
			}
			if status.PathKind != "" {
				turboQuantPathKind = status.PathKind
			}
		}
		if nativeBackendBlocker != "" && fallbackReason == "" {
			fallbackReason = nativeBackendBlocker
		}
		if kvCachePathK == "" {
			kvCachePathK = kvCachePath
		}
		if kvCachePathV == "" {
			kvCachePathV = kvCachePath
		}
	}

	if kvCacheEffectiveK == "" {
		kvCacheEffectiveK = "f16"
	}
	if kvCacheEffectiveV == "" {
		kvCacheEffectiveV = "f16"
	}
	if kvCacheEffective == "" {
		if kvCacheEffectiveK == kvCacheEffectiveV {
			kvCacheEffective = kvCacheEffectiveK
		} else {
			kvCacheEffective = "mixed"
		}
	}
	if kvAlgoResolved == "" {
		if kvAlgoResolvedK == kvAlgoResolvedV {
			kvAlgoResolved = kvAlgoResolvedK
		} else if kvAlgoResolvedK != "" || kvAlgoResolvedV != "" {
			kvAlgoResolved = "mixed"
		}
	}
	if kvCachePath == "" {
		if kvCachePathK == kvCachePathV {
			kvCachePath = kvCachePathK
		} else {
			kvCachePath = "mixed"
		}
	}

	return &InputCache{
		numCtx:                    numCtx,
		enabled:                   cache != nil,
		slots:                     slots,
		multiUserCache:            multiUserCache,
		cache:                     cache,
		kvCacheRequested:          normalizedKVCacheType,
		kvCacheEffective:          kvCacheEffective,
		kvCacheRequestedK:         requestedKVCacheTypeK,
		kvCacheRequestedV:         requestedKVCacheTypeV,
		kvCacheEffectiveK:         kvCacheEffectiveK,
		kvCacheEffectiveV:         kvCacheEffectiveV,
		kvAlgoResolved:            kvAlgoResolved,
		kvAlgoResolvedK:           kvAlgoResolvedK,
		kvAlgoResolvedV:           kvAlgoResolvedV,
		kvCacheBackend:            normalizedKVCacheBackend,
		kvCachePath:               kvCachePath,
		kvCachePathK:              kvCachePathK,
		kvCachePathV:              kvCachePathV,
		kvSymmetric:               kvCacheEffectiveK == kvCacheEffectiveV,
		kvAsymmetric:              kvCacheEffectiveK != kvCacheEffectiveV,
		fallbackReason:            fallbackReason,
		turboQuantPathKind:        turboQuantPathKind,
		nativeTurboQuantActive:    nativeTurboQuantActive,
		referenceTurboQuantActive: referenceTurboQuantActive,
		backendPackedKOwned:       backendPackedKOwned,
		backendPackedVOwned:       backendPackedVOwned,
		backendPackedKAvailable:   backendPackedKAvailable,
		backendPackedVAvailable:   backendPackedVAvailable,
		nativeBackendReady:        nativeBackendReady,
		nativeBackendBlocker:      nativeBackendBlocker,
		vTurboSupported:           vTurboSupported,
		tqBlockSize:               tqBlockSize,
		tqLayoutKind:              tqLayoutKind,
		tqLayoutVersion:           tqLayoutVersion,
		tqGroupCount:              tqGroupCount,
		tqOriginalHeadDim:         tqOriginalHeadDim,
		tqTailPad:                 tqTailPad,
	}, nil
}

func backendTurboQuantSupport(backend ml.Backend) ml.TurboQuantSupport {
	tqBackend, ok := backend.(ml.TurboQuantBackend)
	if !ok {
		return ml.TurboQuantSupport{}
	}
	return tqBackend.TurboQuantSupport()
}

func normalizeKVCacheBackend(s string) string {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "cuda":
		return strings.ToLower(strings.TrimSpace(s))
	default:
		return ""
	}
}

func resolveKVCachePath(backend ml.Backend, dtype ml.DType, requestedBackend string) string {
	if _, ok := kvcachePreset(dtype); !ok {
		return "dense-fallback"
	}

	support := backendTurboQuantSupport(backend)
	if requestedBackend == "cuda" {
		if support.ReferencePackedKCUDA || support.CUDA {
			return "turboquant-cuda-fastpath"
		}
		return "dense-fallback"
	}

	if support.ReferencePackedKCPU || support.CPU {
		return "turboquant-cpu-fastpath"
	}

	return "dense-fallback"
}

func kvCacheTypeFromStr(s string) ml.DType {
	switch normalizeKVCacheType(s) {
	case "q8_0":
		return ml.DTypeQ80
	case "q4_0":
		return ml.DTypeQ40
	case "tq25":
		return ml.DTypeTQ25
	case "tq35":
		return ml.DTypeTQ35
	default:
		return ml.DTypeF16
	}
}

func normalizeKVCacheType(s string) string {
	switch strings.ToLower(s) {
	case "", "off":
		return "f16"
	case "tq3", "tq4":
		return "tq35"
	default:
		return strings.ToLower(s)
	}
}

func resolveKVCacheTypes(unified, k, v string) (string, string) {
	resolvedUnified := normalizeKVCacheType(unified)
	resolvedK := resolvedUnified
	resolvedV := resolvedUnified
	if strings.TrimSpace(k) != "" {
		resolvedK = normalizeKVCacheType(k)
	}
	if strings.TrimSpace(v) != "" {
		resolvedV = normalizeKVCacheType(v)
	}
	if resolvedK == "" {
		resolvedK = "f16"
	}
	if resolvedV == "" {
		resolvedV = "f16"
	}
	return resolvedK, resolvedV
}

func isTurboQuantKVType(s string) bool {
	switch normalizeKVCacheType(s) {
	case "tq25", "tq35":
		return true
	default:
		return false
	}
}

type KVCacheRuntimeInfo struct {
	Requested                 string
	Effective                 string
	RequestedK                string
	RequestedV                string
	EffectiveK                string
	EffectiveV                string
	Algorithm                 string
	AlgorithmK                string
	AlgorithmV                string
	Backend                   string
	Path                      string
	PathK                     string
	PathV                     string
	Symmetric                 bool
	Asymmetric                bool
	FallbackReason            string
	TurboQuantPathKind        string
	NativeTurboQuantActive    bool
	ReferenceTurboQuantActive bool
	BackendPackedKOwned       bool
	BackendPackedVOwned       bool
	BackendPackedKAvailable   bool
	BackendPackedVAvailable   bool
	NativeBackendReady        bool
	NativeBackendBlocker      string
	VTurboSupported           bool
	TQBlockSize               int
	TQLayoutKind              string
	TQLayoutVersion           int
	TQGroupCount              int
	TQOriginalHeadDim         int
	TQTailPad                 int
}

func (c *InputCache) RuntimeInfo() KVCacheRuntimeInfo {
	if c == nil {
		return KVCacheRuntimeInfo{}
	}
	return KVCacheRuntimeInfo{
		Requested:                 c.kvCacheRequested,
		Effective:                 c.kvCacheEffective,
		RequestedK:                c.kvCacheRequestedK,
		RequestedV:                c.kvCacheRequestedV,
		EffectiveK:                c.kvCacheEffectiveK,
		EffectiveV:                c.kvCacheEffectiveV,
		Algorithm:                 c.kvAlgoResolved,
		AlgorithmK:                c.kvAlgoResolvedK,
		AlgorithmV:                c.kvAlgoResolvedV,
		Backend:                   c.kvCacheBackend,
		Path:                      c.kvCachePath,
		PathK:                     c.kvCachePathK,
		PathV:                     c.kvCachePathV,
		Symmetric:                 c.kvSymmetric,
		Asymmetric:                c.kvAsymmetric,
		FallbackReason:            c.fallbackReason,
		TurboQuantPathKind:        c.turboQuantPathKind,
		NativeTurboQuantActive:    c.nativeTurboQuantActive,
		ReferenceTurboQuantActive: c.referenceTurboQuantActive,
		BackendPackedKOwned:       c.backendPackedKOwned,
		BackendPackedVOwned:       c.backendPackedVOwned,
		BackendPackedKAvailable:   c.backendPackedKAvailable,
		BackendPackedVAvailable:   c.backendPackedVAvailable,
		NativeBackendReady:        c.nativeBackendReady,
		NativeBackendBlocker:      c.nativeBackendBlocker,
		VTurboSupported:           c.vTurboSupported,
		TQBlockSize:               c.tqBlockSize,
		TQLayoutKind:              c.tqLayoutKind,
		TQLayoutVersion:           c.tqLayoutVersion,
		TQGroupCount:              c.tqGroupCount,
		TQOriginalHeadDim:         c.tqOriginalHeadDim,
		TQTailPad:                 c.tqTailPad,
	}
}

func kvcachePreset(dtype ml.DType) (turboquant.Preset, bool) {
	switch dtype {
	case ml.DTypeTQ25:
		return turboquant.PresetTQ25, true
	case ml.DTypeTQ35:
		return turboquant.PresetTQ35, true
	default:
		return turboquant.Preset{}, false
	}
}

func (c *InputCache) Close() {
	if c != nil && c.cache != nil {
		c.cache.Close()
	}
}

// Locking: Operations on InputCacheSlot (including finding one
// through LoadCacheSlot) require a lock to be held that serializes
// these operations with each other and processBatch

type InputCacheSlot struct {
	// Index in the KV cache
	Id int

	// Inputs that are stored in the KV cache
	Inputs []*input.Input

	// is this cache actively being processed as part of a sequence?
	InUse bool

	// last time this cache was used (as of start of processing)
	lastUsed time.Time
}

func (c *InputCache) LoadCacheSlot(prompt []*input.Input, cachePrompt bool) (*InputCacheSlot, []*input.Input, error) {
	var slot *InputCacheSlot
	var numPast int32
	var err error

	// In single-user scenarios, the longest cache slot works fine for getting good input
	// cache hit rates and it keeps the footprint of the cache small, which improves throughput.
	// For multiple users, the "best" cache slot produces better input cache hit rates
	// at the cost of worse performance when we miss the input cache.
	if !c.multiUserCache {
		slot, numPast, err = c.findLongestCacheSlot(prompt)
	} else {
		slot, numPast, err = c.findBestCacheSlot(prompt)
	}
	if err != nil {
		return nil, nil, err
	}

	if !cachePrompt {
		numPast = 0
	}

	slot.InUse = true
	slot.lastUsed = time.Now()

	if numPast == int32(len(prompt)) {
		// Leave one input to sample so we can get a response
		numPast--
	}

	if c.cache != nil {
		if numPast > 0 {
			// Recurrent caches use checkpoints to pick a safe resume position.
			if cc, ok := c.cache.(kvcache.CheckpointCache); ok {
				if restored, ok := cc.PrepareRestore(slot.Id, numPast); ok {
					numPast = restored
				} else {
					numPast = 0
				}
			} else if !c.cache.CanResume(slot.Id, numPast) {
				numPast = 0
			}
		}

		err = c.cache.Remove(slot.Id, numPast, math.MaxInt32)
		if err != nil {
			// Some models don't support partial erasure
			err = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			if err != nil {
				return nil, nil, err
			}
			numPast = 0
		}
	}

	slog.Debug("loading cache slot", "id", slot.Id, "cache", len(slot.Inputs), "prompt", len(prompt),
		"used", numPast, "remaining", int32(len(prompt))-numPast)

	slot.Inputs = prompt[:numPast]
	prompt = prompt[numPast:]

	return slot, prompt, nil
}

func (c *InputCache) findLongestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		if s.InUse {
			continue
		}

		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}
	}

	if longestSlot == nil {
		return nil, 0, errors.New("no available cache slots")
	}

	return longestSlot, longest, nil
}

func (c *InputCache) findBestCacheSlot(prompt []*input.Input) (*InputCacheSlot, int32, error) {
	oldest := time.Now()
	var oldestSlot *InputCacheSlot

	longest := int32(-1)
	var longestSlot *InputCacheSlot

	for i, s := range c.slots {
		count := countCommonPrefix(s.Inputs, prompt)
		if count > longest {
			longest = count
			longestSlot = &c.slots[i]
		}

		if s.lastUsed.Compare(oldest) < 0 && !s.InUse {
			oldest = s.lastUsed
			oldestSlot = &c.slots[i]
		}
	}

	if longest == int32(len(longestSlot.Inputs)) && !longestSlot.InUse {
		return longestSlot, longest, nil
	}

	if oldestSlot.InUse {
		return nil, 0, errors.New("no available cache slots")
	}

	if len(oldestSlot.Inputs) != 0 {
		slog.Debug("evicting cache slot", "id", oldestSlot.Id, "inputs", len(oldestSlot.Inputs),
			"used", oldestSlot.lastUsed)
	}

	if longest > 0 && longestSlot != oldestSlot {
		slog.Debug("forking cache slot", "src", longestSlot.Id, "dst", oldestSlot.Id, "inputs", longest, "total",
			len(longestSlot.Inputs))
		oldestSlot.Inputs = make([]*input.Input, longest)
		copy(oldestSlot.Inputs, longestSlot.Inputs[:longest])
		if c.cache != nil {
			c.cache.CopyPrefix(longestSlot.Id, oldestSlot.Id, longest)
		}
	}

	return oldestSlot, longest, nil
}

func countCommonPrefix(a []*input.Input, b []*input.Input) int32 {
	var count int32

	for i := range a {
		if i >= len(b) {
			break
		}

		if a[i].Token != b[i].Token || a[i].MultimodalHash != b[i].MultimodalHash {
			break
		}

		count++
	}

	return count
}

// ShiftDiscard computes how many inputs can be discarded from the cache. Inputs in the same batch
// are discarded together.
func (c *InputCache) ShiftDiscard(inputs []*input.Input, numKeep int32) int32 {
	targetFree := max((c.numCtx-numKeep)/2, 1)
	currentFree := c.numCtx - int32(len(inputs))

	var discard, sameBatch int32
	for _, input := range inputs[numKeep:] {
		if sameBatch <= 0 && currentFree >= targetFree {
			break
		}

		sameBatch--
		currentFree++
		discard++

		if input.SameBatch > 0 {
			sameBatch = int32(input.SameBatch)
		}
	}

	return discard
}

type ErrReprocessInputs struct {
	Inputs []*input.Input
}

func (e *ErrReprocessInputs) Error() string {
	return fmt.Sprintf("kv cache shift not supported, inputs need reprocessing (input count: %v)", len(e.Inputs))
}

// Frees up space in the KV cache by deleting the oldest half of history and shifting
// the newest half into that space (saving numKeep inputs at the beginning).
//
// Assumes that at least 1 entry can be freed up by shifting (i.e. numKeep < numCtx)
func (c *InputCache) ShiftCacheSlot(slot *InputCacheSlot, numKeep int32) error {
	if numKeep >= c.numCtx {
		return fmt.Errorf("unable to shift context - keep exceeds context (keep: %v context: %v)", numKeep, c.numCtx)
	}

	inputLen := int32(len(slot.Inputs))
	discard := c.ShiftDiscard(slot.Inputs, numKeep)

	if discard <= 0 {
		return nil
	}

	slog.Debug("context limit hit - shifting", "id", slot.Id, "limit", c.numCtx, "input", len(slot.Inputs),
		"keep", numKeep, "discard", discard)

	if c.cache != nil {
		err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
		if err != nil {
			slog.Debug("kv cache removal unsupported, clearing cache and returning inputs for reprocessing",
				"id", slot.Id, "error", err)

			// Create new input slice with preserved tokens (numKeep + remaining tokens after discard)
			newInputs := make([]*input.Input, numKeep+inputLen-(numKeep+discard))
			copy(newInputs[:numKeep], slot.Inputs[:numKeep])
			copy(newInputs[numKeep:], slot.Inputs[numKeep+discard:])

			// Reset the cache
			_ = c.cache.Remove(slot.Id, 0, math.MaxInt32)
			slot.Inputs = []*input.Input{}

			// Return error with inputs that need to be reprocessed
			return &ErrReprocessInputs{Inputs: newInputs}
		}
	}

	for i := numKeep + discard; i < inputLen; i++ {
		slot.Inputs[i-discard] = slot.Inputs[i]
	}
	slot.Inputs = slot.Inputs[:inputLen-discard]

	return nil
}
