package ollamarunner

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
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

	kvCacheRequested                string
	kvCacheEffective                string
	kvCacheRequestedK               string
	kvCacheRequestedV               string
	kvCacheEffectiveK               string
	kvCacheEffectiveV               string
	kvAlgoResolved                  string
	kvAlgoResolvedK                 string
	kvAlgoResolvedV                 string
	kvCacheBackend                  string
	kvCachePath                     string
	kvCachePathK                    string
	kvCachePathV                    string
	kvSymmetric                     bool
	kvAsymmetric                    bool
	requestedMode                   string
	effectiveMode                   string
	fallbackReason                  string
	fallbackApplied                 bool
	kOnlyFallback                   bool
	turboQuantPathKind              string
	nativeTurboQuantActive          bool
	referenceTurboQuantActive       bool
	backendPackedKOwned             bool
	backendPackedVOwned             bool
	backendPackedKAvailable         bool
	backendPackedVAvailable         bool
	nativeBackendReady              bool
	nativeBackendBlocker            string
	faEnabled                       bool
	faRequiredForVTurbo             bool
	vTurboSupported                 bool
	detectedHeadDim                 int
	headDimSource                   string
	architectureClass               string
	supportTier                     string
	supportReason                   string
	unsupportedReason               string
	hybridKVArchitecture            bool
	nativeTurboQuantAllowed         bool
	presetRequested                 string
	presetResolved                  string
	presetWarning                   string
	pairingValidated                bool
	experimentalLane                bool
	tqBlockSize                     int
	tqLayoutKind                    string
	tqLayoutVersion                 int
	tqGroupCount                    int
	tqOriginalHeadDim               int
	tqTailPad                       int
	vReconstructionComputeDType     string
	segmentedHeadActive             bool
	segmentedHeadPlan               string
	attentionSurfacePolicyRequested string
	attentionSurfacePolicyEffective string
	attentionSurfaceBehavior        string
	attentionSurfaceOverrideApplied bool
	attentionSurfaceOverrideReason  string
	attentionSurfaceClasses         string
	qjlKRequested                   bool
	qjlVRequested                   bool
	qjlKEnabled                     bool
	qjlVEnabled                     bool
	residualTailTokens              int
	experimentalWeightQuantization  string
	experimentalWeightQuantPolicy   string
	experimentalWeightQuantSource   string
	experimentalWeightQuantActive   bool
}

type turboQuantExperimentalSettings struct {
	QJLKEnabled                    bool
	QJLVEnabled                    bool
	ResidualTailTokens             int
	AllowSegmentedHeads            bool
	SurfacePolicy                  string
	ExperimentalWeightQuantization string
	ExperimentalWeightQuantPolicy  string
	ExperimentalWeightQuantSource  string
}

func NewInputCache(model model.Model, kvCacheType, kvCacheTypeK, kvCacheTypeV, kvCacheBackend string, flashAttention ml.FlashAttentionType, kvSize int32, numSlots int, batchSize int, multiUserCache bool, experimental turboQuantExperimentalSettings) (*InputCache, error) {
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
	fallbackApplied := false
	kOnlyFallback := false
	turboQuantPathKind := "disabled"
	nativeTurboQuantActive := false
	referenceTurboQuantActive := false
	backendPackedKOwned := false
	backendPackedVOwned := false
	backendPackedKAvailable := false
	backendPackedVAvailable := false
	nativeBackendReady := false
	nativeBackendBlocker := ""
	faEnabled := flashAttention == ml.FlashAttentionEnabled
	faRequiredForVTurbo := false
	vTurboSupported := normalizedKVCacheTypeV == "" || normalizedKVCacheTypeV == "f16"
	support := detectTurboQuantModelSupport(model)
	detectedHeadDim := support.DetectedHeadDim
	headDimSource := string(support.HeadDimSource)
	architectureClass := support.ArchitectureClass
	supportTier := string(support.SupportTier)
	supportReason := support.SupportReason
	unsupportedReason := support.UnsupportedReason
	hybridKVArchitecture := support.HybridKVArchitecture
	nativeTurboQuantAllowed := support.NativeTurboQuantAllowed
	presetRequested := string(normalizeTurboQuantRolloutPreset(envconfig.TurboQuantPreset()))
	presetResolved := ""
	presetWarning := ""
	pairingValidated := false
	experimentalLane := false
	tqBlockSize := 0
	tqLayoutKind := ""
	tqLayoutVersion := 0
	tqGroupCount := 0
	tqOriginalHeadDim := 0
	tqTailPad := 0
	vReconstructionComputeDType := ""
	segmentedHeadActive := false
	segmentedHeadPlan := ""
	attentionSurfacePolicyRequested := normalizeAttentionSurfacePolicy(experimental.SurfacePolicy)
	attentionSurfacePolicyEffective := attentionSurfacePolicyRequested
	if attentionSurfacePolicyEffective == "auto" {
		attentionSurfacePolicyEffective = support.AttentionSurfacePolicy
		if attentionSurfacePolicyEffective == "" {
			attentionSurfacePolicyEffective = "uniform"
		}
	}
	attentionSurfaceBehavior := "global=f16"
	attentionSurfaceOverrideApplied := false
	attentionSurfaceOverrideReason := ""
	attentionSurfaceClasses := strings.Join(support.AttentionSurfaceClasses, ",")
	qjlKRequested := experimental.QJLKEnabled
	qjlVRequested := experimental.QJLVEnabled
	qjlKEnabled := experimental.QJLKEnabled
	qjlVEnabled := experimental.QJLVEnabled
	residualTailTokens := max(0, experimental.ResidualTailTokens)
	experimentalWeightQuantization := normalizeExperimentalWeightQuantization(experimental.ExperimentalWeightQuantization)
	experimentalWeightQuantPolicy := normalizeExperimentalWeightQuantPolicy(experimental.ExperimentalWeightQuantPolicy)
	experimentalWeightQuantSource := normalizeExperimentalWeightQuantSource(experimental.ExperimentalWeightQuantSource)
	experimentalWeightQuantActive := false

	if qjlVEnabled {
		qjlVEnabled = false
		fallbackApplied = true
		fallbackReason = appendFallbackReason(fallbackReason, "V-side QJL is experimental-disabled")
	}
	if qjlKEnabled && !support.QJLKAllowed {
		qjlKEnabled = false
		fallbackApplied = true
		fallbackReason = appendFallbackReason(fallbackReason, "K-side QJL is not allowlisted for this model family/support tier")
	}
	if experimentalWeightQuantization != "off" {
		if !support.ExperimentalWeightQuantAllowed {
			fallbackApplied = true
			fallbackReason = appendFallbackReason(fallbackReason, "experimental weight quantization is not allowlisted for this model family/support tier")
		} else {
			// Implemented scaffold-only weight quant lane metadata so KV and weight compression stay separate; idea source: @animehacker.
			experimentalLane = true
			fallbackReason = appendFallbackReason(fallbackReason, "experimental weight quantization is scaffold-only; no active model-weight compression applied")
		}
	}
	if attentionSurfacePolicyEffective == "surface-aware" && support.SWABypassRecommended && (isTurboQuantKVType(requestedKVCacheTypeK) || isTurboQuantKVType(requestedKVCacheTypeV)) {
		// Implemented surface-aware whole-model fallback when per-surface dispatch is unavailable, so SWA-heavy models are not silently quantized uniformly; idea source: @AmesianX.
		attentionSurfaceOverrideApplied = true
		attentionSurfaceOverrideReason = "surface-aware policy requested but per-surface dispatch is unavailable; falling back to f16 for fragile attention surfaces"
		attentionSurfacePolicyEffective = "uniform"
		attentionSurfaceBehavior = "global=f16,swa=f16,dispatch=whole-model-fallback"
		fallbackApplied = true
		fallbackReason = appendFallbackReason(fallbackReason, attentionSurfaceOverrideReason)
		normalizedKVCacheTypeK = "f16"
		normalizedKVCacheTypeV = "f16"
	}
	if experimental.AllowSegmentedHeads && support.SegmentedHeadExperimental {
		segmentedHeadActive = true
		segmentedHeadPlan = joinIntSlice(support.SegmentedHeadPlan, "+")
		experimentalLane = true
	}
	if cache != nil {
		backendSupport := backendTurboQuantSupport(model.Backend())
		dtype := kvCacheTypeFromStr(normalizedKVCacheType)
		dtypeK := kvCacheTypeFromStr(normalizedKVCacheTypeK)
		dtypeV := kvCacheTypeFromStr(normalizedKVCacheTypeV)
		kvCachePathK = resolveKVCachePath(model.Backend(), dtypeK, normalizedKVCacheBackend)
		kvCachePathV = resolveKVCachePath(model.Backend(), dtypeV, normalizedKVCacheBackend)
		normalizedKVCacheTypeK, normalizedKVCacheTypeV, kvCacheEffectiveK, kvCacheEffectiveV, fallbackReason, fallbackApplied, kOnlyFallback, vTurboSupported, faRequiredForVTurbo = resolveTurboQuantFallback(
			backendSupport,
			normalizedKVCacheBackend,
			faEnabled,
			normalizedKVCacheTypeK,
			normalizedKVCacheTypeV,
			kvCachePathK,
			kvCachePathV,
		)
		dtypeK = kvCacheTypeFromStr(normalizedKVCacheTypeK)
		dtypeV = kvCacheTypeFromStr(normalizedKVCacheTypeV)
		kvCachePathK = resolveKVCachePath(model.Backend(), dtypeK, normalizedKVCacheBackend)
		kvCachePathV = resolveKVCachePath(model.Backend(), dtypeV, normalizedKVCacheBackend)
		if isTurboQuantKVType(normalizedKVCacheTypeK) && kvCachePathK != "dense-fallback" {
			kvAlgoResolvedK = turboquant.AlgorithmPaper
		}
		if isTurboQuantKVType(normalizedKVCacheTypeV) && kvCachePathV != "dense-fallback" {
			kvAlgoResolvedV = turboquant.AlgorithmPaper
			vTurboSupported = true
		}
		if preset, ok := kvcachePreset(dtype); ok {
			cache = kvcache.WrapWithTurboQuant(cache, preset, normalizedKVCacheBackend)
			kvcache.ConfigureTurboQuantExperimental(cache, kvcache.TurboQuantExperimentalConfig{
				QJLKEnabled:                 qjlKEnabled,
				QJLVEnabled:                 qjlVEnabled,
				ResidualTailTokens:          residualTailTokens,
				SegmentedHeadActive:         segmentedHeadActive,
				SegmentedHeadPlan:           turboquant.SegmentedHeadDimPlan{HeadDim: support.DetectedHeadDim, Segments: append([]int(nil), support.SegmentedHeadPlan...), Experimental: support.SegmentedHeadExperimental},
				VReconstructionComputeDType: "fp32",
			})
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
		if !nativeTurboQuantAllowed {
			nativeBackendReady = false
			if nativeBackendBlocker == "" {
				nativeBackendBlocker = formatNativeSupportFallbackReason(support, referenceTurboQuantActive)
			}
		}
		if nativeBackendBlocker != "" && fallbackReason == "" {
			fallbackReason = nativeBackendBlocker
		}
		nativeFallbackReason := ""
		if !nativeTurboQuantAllowed && (turboQuantPathKind == "native_grouped_scaffold" || turboQuantPathKind == "native_backend") {
			nativeFallbackReason = formatNativeSupportFallbackReason(support, referenceTurboQuantActive)
			if referenceTurboQuantActive {
				turboQuantPathKind = "reference_wrapper"
				nativeTurboQuantActive = false
				nativeBackendReady = false
				nativeBackendBlocker = nativeFallbackReason
			} else {
				if isTurboQuantKVType(kvCacheEffectiveK) {
					kvCacheEffectiveK = "f16"
					kvAlgoResolvedK = ""
					kvCachePathK = "dense-fallback"
				}
				if isTurboQuantKVType(kvCacheEffectiveV) {
					kvCacheEffectiveV = "f16"
					kvAlgoResolvedV = ""
					kvCachePathV = "dense-fallback"
					vTurboSupported = false
				}
				if kvCacheEffectiveK == kvCacheEffectiveV {
					kvCacheEffective = kvCacheEffectiveK
				} else {
					kvCacheEffective = "mixed"
				}
				turboQuantPathKind = "disabled"
				nativeTurboQuantActive = false
				referenceTurboQuantActive = false
				nativeBackendReady = false
				nativeBackendBlocker = nativeFallbackReason
			}
			fallbackApplied = true
			if fallbackReason == "" {
				fallbackReason = nativeFallbackReason
			} else if !strings.Contains(fallbackReason, nativeFallbackReason) {
				fallbackReason = fallbackReason + "; " + nativeFallbackReason
			}
		}
		if fallbackReason != "" {
			slog.Warn("falling back from requested V-side turboquant mode",
				"requested_k_type", requestedKVCacheTypeK,
				"requested_v_type", requestedKVCacheTypeV,
				"effective_k_type", kvCacheEffectiveK,
				"effective_v_type", kvCacheEffectiveV,
				"backend", normalizedKVCacheBackend,
				"fa_enabled", faEnabled,
				"fa_required_for_v_turbo", faRequiredForVTurbo,
				"reason", fallbackReason,
			)
		}
		slog.Info("turboquant runtime support",
			"requested_k_type", requestedKVCacheTypeK,
			"requested_v_type", requestedKVCacheTypeV,
			"effective_k_type", kvCacheEffectiveK,
			"effective_v_type", kvCacheEffectiveV,
			"detected_head_dim", detectedHeadDim,
			"head_dim_source", headDimSource,
			"architecture_class", architectureClass,
			"support_tier", supportTier,
			"hybrid_kv_architecture", hybridKVArchitecture,
			"native_turboquant_allowed", nativeTurboQuantAllowed,
			"turboquant_path_kind", turboQuantPathKind,
			"fallback_reason", fallbackReason,
		)
		if kvCachePathK == "" {
			kvCachePathK = kvCachePath
		}
		if kvCachePathV == "" {
			kvCachePathV = kvCachePath
		}
	}

	if isTurboQuantKVType(kvCacheEffectiveV) {
		vReconstructionComputeDType = effectiveVReconstructionComputeDType(turboQuantPathKind, backendPackedVOwned, backendSupportFor(model.Backend()), referenceTurboQuantActive)
		if vReconstructionComputeDType != "fp32" {
			fallbackApplied = true
			if isTurboQuantKVType(kvCacheEffectiveV) {
				kvCacheEffectiveV = "f16"
				kvCachePathV = "dense-fallback"
				kvAlgoResolvedV = ""
				kOnlyFallback = isTurboQuantKVType(kvCacheEffectiveK)
				fallbackReason = appendFallbackReason(fallbackReason, "supported V reconstruction path did not guarantee fp32 compute")
			}
			if kvCacheEffectiveK == kvCacheEffectiveV {
				kvCacheEffective = kvCacheEffectiveK
			} else {
				kvCacheEffective = "mixed"
			}
		}
	} else if isTurboQuantKVType(kvCacheEffectiveK) {
		vReconstructionComputeDType = "fp32"
	}
	if attentionSurfaceBehavior == "global=f16" {
		attentionSurfaceBehavior = deriveAttentionSurfaceBehavior(attentionSurfaceClasses, attentionSurfaceOverrideApplied, kvCacheEffectiveK, kvCacheEffectiveV)
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
	requestedMode := summarizeKVMode(requestedKVCacheTypeK, requestedKVCacheTypeV)
	effectiveMode := summarizeKVMode(kvCacheEffectiveK, kvCacheEffectiveV)
	// Implemented rollout-preset guidance so recommendation lanes stay tied to validated mixed-K/V data; idea source: @seanrasch.
	// Implemented the conservative asymmetric recommendation lane around q8_0-K plus TurboQuant V when validation supports it; idea source: @primoco.
	// Implemented preset warnings so asymmetric and experimental pairings stay guarded instead of being implied production-safe; idea source: @sjoerdmaessen.
	recommendation := resolveTurboQuantPreset(presetRequested, support, requestedKVCacheTypeK, requestedKVCacheTypeV, kvCacheEffectiveK, kvCacheEffectiveV, turboQuantPathKind, fallbackReason, faEnabled, vTurboSupported)
	presetResolved = string(recommendation.Preset)
	presetWarning = recommendation.Warning
	pairingValidated = recommendation.PairingValidated
	experimentalLane = recommendation.ExperimentalLane
	if presetWarning != "" {
		slog.Warn("turboquant rollout preset warning",
			"preset_requested", presetRequested,
			"preset_resolved", presetResolved,
			"pairing_validated", pairingValidated,
			"experimental_lane", experimentalLane,
			"warning", presetWarning,
		)
	}

	return &InputCache{
		numCtx:                          numCtx,
		enabled:                         cache != nil,
		slots:                           slots,
		multiUserCache:                  multiUserCache,
		cache:                           cache,
		kvCacheRequested:                normalizedKVCacheType,
		kvCacheEffective:                kvCacheEffective,
		kvCacheRequestedK:               requestedKVCacheTypeK,
		kvCacheRequestedV:               requestedKVCacheTypeV,
		kvCacheEffectiveK:               kvCacheEffectiveK,
		kvCacheEffectiveV:               kvCacheEffectiveV,
		kvAlgoResolved:                  kvAlgoResolved,
		kvAlgoResolvedK:                 kvAlgoResolvedK,
		kvAlgoResolvedV:                 kvAlgoResolvedV,
		kvCacheBackend:                  normalizedKVCacheBackend,
		kvCachePath:                     kvCachePath,
		kvCachePathK:                    kvCachePathK,
		kvCachePathV:                    kvCachePathV,
		kvSymmetric:                     kvCacheEffectiveK == kvCacheEffectiveV,
		kvAsymmetric:                    kvCacheEffectiveK != kvCacheEffectiveV,
		requestedMode:                   requestedMode,
		effectiveMode:                   effectiveMode,
		fallbackReason:                  fallbackReason,
		fallbackApplied:                 fallbackApplied || requestedMode != effectiveMode,
		kOnlyFallback:                   kOnlyFallback,
		turboQuantPathKind:              turboQuantPathKind,
		nativeTurboQuantActive:          nativeTurboQuantActive,
		referenceTurboQuantActive:       referenceTurboQuantActive,
		backendPackedKOwned:             backendPackedKOwned,
		backendPackedVOwned:             backendPackedVOwned,
		backendPackedKAvailable:         backendPackedKAvailable,
		backendPackedVAvailable:         backendPackedVAvailable,
		nativeBackendReady:              nativeBackendReady,
		nativeBackendBlocker:            nativeBackendBlocker,
		faEnabled:                       faEnabled,
		faRequiredForVTurbo:             faRequiredForVTurbo,
		vTurboSupported:                 vTurboSupported,
		detectedHeadDim:                 detectedHeadDim,
		headDimSource:                   headDimSource,
		architectureClass:               architectureClass,
		supportTier:                     supportTier,
		supportReason:                   supportReason,
		unsupportedReason:               unsupportedReason,
		hybridKVArchitecture:            hybridKVArchitecture,
		nativeTurboQuantAllowed:         nativeTurboQuantAllowed,
		presetRequested:                 presetRequested,
		presetResolved:                  presetResolved,
		presetWarning:                   presetWarning,
		pairingValidated:                pairingValidated,
		experimentalLane:                experimentalLane,
		tqBlockSize:                     tqBlockSize,
		tqLayoutKind:                    tqLayoutKind,
		tqLayoutVersion:                 tqLayoutVersion,
		tqGroupCount:                    tqGroupCount,
		tqOriginalHeadDim:               tqOriginalHeadDim,
		tqTailPad:                       tqTailPad,
		vReconstructionComputeDType:     vReconstructionComputeDType,
		segmentedHeadActive:             segmentedHeadActive,
		segmentedHeadPlan:               segmentedHeadPlan,
		attentionSurfacePolicyRequested: attentionSurfacePolicyRequested,
		attentionSurfacePolicyEffective: attentionSurfacePolicyEffective,
		attentionSurfaceBehavior:        attentionSurfaceBehavior,
		attentionSurfaceOverrideApplied: attentionSurfaceOverrideApplied,
		attentionSurfaceOverrideReason:  attentionSurfaceOverrideReason,
		attentionSurfaceClasses:         attentionSurfaceClasses,
		qjlKRequested:                   qjlKRequested,
		qjlVRequested:                   qjlVRequested,
		qjlKEnabled:                     qjlKEnabled,
		qjlVEnabled:                     qjlVEnabled,
		residualTailTokens:              residualTailTokens,
		experimentalWeightQuantization:  experimentalWeightQuantization,
		experimentalWeightQuantPolicy:   experimentalWeightQuantPolicy,
		experimentalWeightQuantSource:   experimentalWeightQuantSource,
		experimentalWeightQuantActive:   experimentalWeightQuantActive,
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

func requiresFlashAttentionForVTurbo(support ml.TurboQuantSupport, requestedV ml.DType) bool {
	if requestedV != ml.DTypeTQ25 && requestedV != ml.DTypeTQ35 {
		return false
	}
	return support.RequiresFlashAttention
}

func backendPackedVSupportedForBackend(support ml.TurboQuantSupport, requestedBackend string) bool {
	switch strings.ToLower(strings.TrimSpace(requestedBackend)) {
	case "cuda":
		return support.BackendPackedVCUDA
	default:
		return support.BackendPackedVCPU
	}
}

func resolveTurboQuantFallback(support ml.TurboQuantSupport, requestedBackend string, faEnabled bool, requestedKType, requestedVType, pathK, pathV string) (effectiveKType, effectiveVType, resolvedKType, resolvedVType string, fallbackReason string, fallbackApplied bool, kOnlyFallback bool, vTurboSupported bool, faRequiredForVTurbo bool) {
	effectiveKType = requestedKType
	effectiveVType = requestedVType
	resolvedKType = requestedKType
	resolvedVType = requestedVType
	vTurboSupported = !isTurboQuantKVType(requestedVType)
	faRequiredForVTurbo = requiresFlashAttentionForVTurbo(support, kvCacheTypeFromStr(requestedVType))

	kSupported := !isTurboQuantKVType(requestedKType) || pathK != "dense-fallback"
	vPathSupported := !isTurboQuantKVType(requestedVType) || pathV != "dense-fallback"

	if !isTurboQuantKVType(requestedVType) {
		return
	}

	// Implemented V-side TurboQuant gating on Flash Attention availability for the active backend; idea source: @Madreag.
	if faRequiredForVTurbo && !faEnabled {
		fallbackApplied = true
		fallbackReason = "requested V turboquant requires Flash Attention on the active backend; falling back to f16 on V"
		effectiveVType = "f16"
		resolvedVType = "f16"
		vTurboSupported = false
		if kSupported && isTurboQuantKVType(requestedKType) {
			// Implemented explicit K-only fallback reporting so unsupported V turbo requests do not degrade ambiguously; idea source: @TheTom.
			kOnlyFallback = true
			return
		}
		effectiveKType = "f16"
		resolvedKType = "f16"
		return
	}

	if !vPathSupported {
		fallbackApplied = true
		fallbackReason = fmt.Sprintf("requested V turboquant path is not supported on backend=%s path=%s; falling back to f16 on V", firstNonEmpty(requestedBackend, "cpu"), pathV)
		effectiveVType = "f16"
		resolvedVType = "f16"
		vTurboSupported = false
		if kSupported && isTurboQuantKVType(requestedKType) {
			kOnlyFallback = true
			return
		}
		effectiveKType = "f16"
		resolvedKType = "f16"
		return
	}

	if backendPackedVSupportedForBackend(support, requestedBackend) && strings.ToLower(strings.TrimSpace(support.VReconstructionComputeDType)) != "fp32" {
		fallbackApplied = true
		fallbackReason = "requested V turboquant backend path does not guarantee fp32 reconstruction; falling back to f16 on V"
		effectiveVType = "f16"
		resolvedVType = "f16"
		vTurboSupported = false
		if kSupported && isTurboQuantKVType(requestedKType) {
			kOnlyFallback = true
			return
		}
		effectiveKType = "f16"
		resolvedKType = "f16"
		return
	}

	vTurboSupported = true
	return
}

func summarizeKVMode(kType, vType string) string {
	kType = strings.TrimSpace(kType)
	if kType == "" {
		kType = "f16"
	}
	vType = strings.TrimSpace(vType)
	if vType == "" {
		vType = "f16"
	}
	if kType == vType {
		return kType
	}
	return fmt.Sprintf("k=%s,v=%s", kType, vType)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func formatNativeSupportFallbackReason(support turboQuantModelSupport, referenceWrapperAvailable bool) string {
	target := "reference wrapper"
	if !referenceWrapperAvailable {
		target = "f16 because reference wrapper is unavailable"
	}
	switch {
	case support.UnsupportedReason != "" && support.DetectedHeadDim > 0:
		return fmt.Sprintf("native TurboQuant disabled: %s; falling back to %s", support.UnsupportedReason, target)
	case support.SupportReason != "" && support.HybridKVArchitecture:
		return fmt.Sprintf("native TurboQuant disabled: %s; falling back to %s", support.SupportReason, target)
	case support.UnsupportedReason != "":
		return fmt.Sprintf("native TurboQuant disabled: %s; falling back to %s", support.UnsupportedReason, target)
	default:
		return fmt.Sprintf("native TurboQuant disabled by support matrix; falling back to %s", target)
	}
}

type KVCacheRuntimeInfo struct {
	Requested                       string
	Effective                       string
	RequestedK                      string
	RequestedV                      string
	EffectiveK                      string
	EffectiveV                      string
	RequestedMode                   string
	EffectiveMode                   string
	Algorithm                       string
	AlgorithmK                      string
	AlgorithmV                      string
	Backend                         string
	Path                            string
	PathK                           string
	PathV                           string
	Symmetric                       bool
	Asymmetric                      bool
	FallbackReason                  string
	FallbackApplied                 bool
	KOnlyFallback                   bool
	TurboQuantPathKind              string
	NativeTurboQuantActive          bool
	ReferenceTurboQuantActive       bool
	BackendPackedKOwned             bool
	BackendPackedVOwned             bool
	BackendPackedKAvailable         bool
	BackendPackedVAvailable         bool
	NativeBackendReady              bool
	NativeBackendBlocker            string
	FAEnabled                       bool
	FARequiredForVTurbo             bool
	VTurboSupported                 bool
	DetectedHeadDim                 int
	HeadDimSource                   string
	ArchitectureClass               string
	SupportTier                     string
	SupportReason                   string
	UnsupportedReason               string
	HybridKVArchitecture            bool
	NativeTurboQuantAllowed         bool
	PresetRequested                 string
	PresetResolved                  string
	PresetWarning                   string
	PairingValidated                bool
	ExperimentalLane                bool
	TQBlockSize                     int
	TQLayoutKind                    string
	TQLayoutVersion                 int
	TQGroupCount                    int
	TQOriginalHeadDim               int
	TQTailPad                       int
	VReconstructionComputeDType     string
	SegmentedHeadActive             bool
	SegmentedHeadPlan               string
	AttentionSurfacePolicyRequested string
	AttentionSurfacePolicyEffective string
	AttentionSurfaceBehavior        string
	AttentionSurfaceOverrideApplied bool
	AttentionSurfaceOverrideReason  string
	AttentionSurfaceClasses         string
	QJLKRequested                   bool
	QJLVRequested                   bool
	QJLKEnabled                     bool
	QJLVEnabled                     bool
	ResidualTailTokens              int
	ExperimentalWeightQuantization  string
	ExperimentalWeightQuantPolicy   string
	ExperimentalWeightQuantSource   string
	ExperimentalWeightQuantActive   bool
}

func (c *InputCache) RuntimeInfo() KVCacheRuntimeInfo {
	if c == nil {
		return KVCacheRuntimeInfo{}
	}
	return KVCacheRuntimeInfo{
		Requested:                       c.kvCacheRequested,
		Effective:                       c.kvCacheEffective,
		RequestedK:                      c.kvCacheRequestedK,
		RequestedV:                      c.kvCacheRequestedV,
		EffectiveK:                      c.kvCacheEffectiveK,
		EffectiveV:                      c.kvCacheEffectiveV,
		RequestedMode:                   c.requestedMode,
		EffectiveMode:                   c.effectiveMode,
		Algorithm:                       c.kvAlgoResolved,
		AlgorithmK:                      c.kvAlgoResolvedK,
		AlgorithmV:                      c.kvAlgoResolvedV,
		Backend:                         c.kvCacheBackend,
		Path:                            c.kvCachePath,
		PathK:                           c.kvCachePathK,
		PathV:                           c.kvCachePathV,
		Symmetric:                       c.kvSymmetric,
		Asymmetric:                      c.kvAsymmetric,
		FallbackReason:                  c.fallbackReason,
		FallbackApplied:                 c.fallbackApplied,
		KOnlyFallback:                   c.kOnlyFallback,
		TurboQuantPathKind:              c.turboQuantPathKind,
		NativeTurboQuantActive:          c.nativeTurboQuantActive,
		ReferenceTurboQuantActive:       c.referenceTurboQuantActive,
		BackendPackedKOwned:             c.backendPackedKOwned,
		BackendPackedVOwned:             c.backendPackedVOwned,
		BackendPackedKAvailable:         c.backendPackedKAvailable,
		BackendPackedVAvailable:         c.backendPackedVAvailable,
		NativeBackendReady:              c.nativeBackendReady,
		NativeBackendBlocker:            c.nativeBackendBlocker,
		FAEnabled:                       c.faEnabled,
		FARequiredForVTurbo:             c.faRequiredForVTurbo,
		VTurboSupported:                 c.vTurboSupported,
		DetectedHeadDim:                 c.detectedHeadDim,
		HeadDimSource:                   c.headDimSource,
		ArchitectureClass:               c.architectureClass,
		SupportTier:                     c.supportTier,
		SupportReason:                   c.supportReason,
		UnsupportedReason:               c.unsupportedReason,
		HybridKVArchitecture:            c.hybridKVArchitecture,
		NativeTurboQuantAllowed:         c.nativeTurboQuantAllowed,
		PresetRequested:                 c.presetRequested,
		PresetResolved:                  c.presetResolved,
		PresetWarning:                   c.presetWarning,
		PairingValidated:                c.pairingValidated,
		ExperimentalLane:                c.experimentalLane,
		TQBlockSize:                     c.tqBlockSize,
		TQLayoutKind:                    c.tqLayoutKind,
		TQLayoutVersion:                 c.tqLayoutVersion,
		TQGroupCount:                    c.tqGroupCount,
		TQOriginalHeadDim:               c.tqOriginalHeadDim,
		TQTailPad:                       c.tqTailPad,
		VReconstructionComputeDType:     c.vReconstructionComputeDType,
		SegmentedHeadActive:             c.segmentedHeadActive,
		SegmentedHeadPlan:               c.segmentedHeadPlan,
		AttentionSurfacePolicyRequested: c.attentionSurfacePolicyRequested,
		AttentionSurfacePolicyEffective: c.attentionSurfacePolicyEffective,
		AttentionSurfaceBehavior:        c.attentionSurfaceBehavior,
		AttentionSurfaceOverrideApplied: c.attentionSurfaceOverrideApplied,
		AttentionSurfaceOverrideReason:  c.attentionSurfaceOverrideReason,
		AttentionSurfaceClasses:         c.attentionSurfaceClasses,
		QJLKRequested:                   c.qjlKRequested,
		QJLVRequested:                   c.qjlVRequested,
		QJLKEnabled:                     c.qjlKEnabled,
		QJLVEnabled:                     c.qjlVEnabled,
		ResidualTailTokens:              c.residualTailTokens,
		ExperimentalWeightQuantization:  c.experimentalWeightQuantization,
		ExperimentalWeightQuantPolicy:   c.experimentalWeightQuantPolicy,
		ExperimentalWeightQuantSource:   c.experimentalWeightQuantSource,
		ExperimentalWeightQuantActive:   c.experimentalWeightQuantActive,
	}
}

func normalizeAttentionSurfacePolicy(policy string) string {
	switch strings.ToLower(strings.TrimSpace(policy)) {
	case "", "auto":
		return "auto"
	case "uniform":
		return "uniform"
	case "surface-aware":
		return "surface-aware"
	default:
		return "auto"
	}
}

func backendSupportFor(backend ml.Backend) ml.TurboQuantSupport {
	if tqBackend, ok := backend.(ml.TurboQuantBackend); ok {
		return tqBackend.TurboQuantSupport()
	}
	return ml.TurboQuantSupport{}
}

func effectiveVReconstructionComputeDType(pathKind string, backendPackedVOwned bool, support ml.TurboQuantSupport, referenceWrapperActive bool) string {
	if referenceWrapperActive || !backendPackedVOwned || pathKind == "reference_wrapper" || pathKind == "dense-fallback" || pathKind == "native_grouped_scaffold" {
		return "fp32"
	}
	return strings.ToLower(strings.TrimSpace(support.VReconstructionComputeDType))
}

func deriveAttentionSurfaceBehavior(classes string, overrideApplied bool, effectiveKType, effectiveVType string) string {
	if overrideApplied {
		return "global=f16,swa=f16,dispatch=whole-model-fallback"
	}
	lowerClasses := strings.ToLower(strings.TrimSpace(classes))
	switch {
	case strings.Contains(lowerClasses, "swa"):
		return fmt.Sprintf("global=%s/%s,swa=f16/f16,dispatch=metadata-only", effectiveKType, effectiveVType)
	case strings.Contains(lowerClasses, "hybrid_unknown"):
		return "hybrid_unknown=f16/f16,dispatch=whole-model-fallback"
	default:
		return fmt.Sprintf("global=%s/%s", effectiveKType, effectiveVType)
	}
}

func normalizeExperimentalWeightQuantization(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "off":
		return "off"
	case "attention_only":
		return "attention_only"
	default:
		return "off"
	}
}

func normalizeExperimentalWeightQuantPolicy(policy string) string {
	switch strings.ToLower(strings.TrimSpace(policy)) {
	case "", "attention_only_safe":
		return "attention_only_safe"
	default:
		return "attention_only_safe"
	}
}

func normalizeExperimentalWeightQuantSource(source string) string {
	switch strings.ToLower(strings.TrimSpace(source)) {
	case "", "f16":
		return "f16"
	case "bf16":
		return "bf16"
	default:
		return "f16"
	}
}

func appendFallbackReason(existing, reason string) string {
	if reason == "" {
		return existing
	}
	if existing == "" {
		return reason
	}
	if strings.Contains(existing, reason) {
		return existing
	}
	return existing + "; " + reason
}

func joinIntSlice(values []int, sep string) string {
	if len(values) == 0 {
		return ""
	}
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, fmt.Sprintf("%d", value))
	}
	return strings.Join(parts, sep)
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
