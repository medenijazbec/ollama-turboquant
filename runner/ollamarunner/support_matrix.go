package ollamarunner

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/model"
)

type turboQuantSupportTier string

const (
	turboQuantSupportSafe         turboQuantSupportTier = "safe"
	turboQuantSupportConservative turboQuantSupportTier = "conservative"
	turboQuantSupportExperimental turboQuantSupportTier = "experimental"
	turboQuantSupportUnsupported  turboQuantSupportTier = "unsupported"
)

type turboQuantHeadDimSource string

const (
	turboQuantHeadDimSourceUnknown              turboQuantHeadDimSource = "unknown"
	turboQuantHeadDimSourceAttentionKeyLength   turboQuantHeadDimSource = "attention.key_length"
	turboQuantHeadDimSourceAttentionValueLength turboQuantHeadDimSource = "attention.value_length"
	turboQuantHeadDimSourceEmbeddingHeadCountKV turboQuantHeadDimSource = "embedding_length/head_count_kv"
	turboQuantHeadDimSourceEmbeddingHeadCount   turboQuantHeadDimSource = "embedding_length/head_count"
	turboQuantHeadDimSourceLayerHeadCountArray  turboQuantHeadDimSource = "layer_head_count_array"
	turboQuantHeadDimSourceRopeDimensionCount   turboQuantHeadDimSource = "rope.dimension_count"
)

type turboQuantModelSupport struct {
	DetectedHeadDim         int
	HeadDimSource           turboQuantHeadDimSource
	ArchitectureClass       string
	SupportTier             turboQuantSupportTier
	SupportReason           string
	UnsupportedReason       string
	HybridKVArchitecture    bool
	NativeTurboQuantAllowed bool
}

func detectTurboQuantModelSupport(m model.Model) turboQuantModelSupport {
	cfg := m.Backend().Config()
	architectureClass := normalizeArchitectureClass(cfg.Architecture(), cfg)
	hybrid := detectHybridKVArchitecture(m, cfg)

	// Implemented multi-stage head-dim detection so native rollout does not depend on a single brittle metadata source; idea source: @AmesianX.
	headDim, source := detectHeadDim(cfg)
	support := turboQuantModelSupport{
		DetectedHeadDim:         headDim,
		HeadDimSource:           source,
		ArchitectureClass:       architectureClass,
		HybridKVArchitecture:    hybrid,
		SupportTier:             turboQuantSupportUnsupported,
		NativeTurboQuantAllowed: false,
	}

	if headDim <= 0 {
		support.UnsupportedReason = "unable to infer attention head dimension from model metadata"
		return support
	}

	// Implemented support-tier gating for unsupported head dimensions and architecture families before native activation; idea source: @fritolays.
	switch architectureClass {
	case "llama-family", "mistral-family", "qwen-family", "gemma-family", "phi-family":
		switch {
		case headDim == 128:
			support.SupportTier = turboQuantSupportSafe
			support.SupportReason = "head_dim=128 is in the primary validated rollout lane"
			support.NativeTurboQuantAllowed = !hybrid
		case headDim == 64 || headDim == 96:
			support.SupportTier = turboQuantSupportConservative
			support.SupportReason = fmt.Sprintf("head_dim=%d is supported in the conservative rollout lane", headDim)
			support.NativeTurboQuantAllowed = !hybrid
		case headDim == 256:
			support.SupportTier = turboQuantSupportExperimental
			support.SupportReason = "head_dim=256 is experimental and requires explicit rollout controls"
			support.NativeTurboQuantAllowed = !hybrid
		case headDim%32 != 0:
			support.UnsupportedReason = fmt.Sprintf("unsupported head_dim=%d for native TurboQuant rollout", headDim)
		default:
			support.SupportTier = turboQuantSupportConservative
			support.SupportReason = fmt.Sprintf("head_dim=%d is aligned but outside the primary validated lane", headDim)
			support.NativeTurboQuantAllowed = !hybrid
		}
	case "gptoss-hybrid", "sliding-window-hybrid":
		if headDim%32 == 0 {
			support.SupportTier = turboQuantSupportExperimental
			support.SupportReason = "hybrid KV handling is detectable but not yet promoted to native rollout"
		} else {
			support.SupportTier = turboQuantSupportUnsupported
			support.UnsupportedReason = fmt.Sprintf("hybrid KV architecture with unsupported head_dim=%d", headDim)
		}
	case "deepseek-mla", "glm-mla":
		support.UnsupportedReason = "MLA / compressed-KV topology is provisional and not yet supported for native TurboQuant activation"
	case "bert-like", "vision-only", "unknown":
		support.UnsupportedReason = fmt.Sprintf("unsupported architecture_class=%s for native TurboQuant rollout", architectureClass)
	default:
		support.UnsupportedReason = fmt.Sprintf("unsupported architecture_class=%s for native TurboQuant rollout", architectureClass)
	}

	if hybrid {
		support.HybridKVArchitecture = true
		if support.SupportReason == "" {
			support.SupportReason = "hybrid KV architecture requires wrapper/reference handling in this rollout stage"
		}
		support.NativeTurboQuantAllowed = false
		if support.UnsupportedReason == "" && support.SupportTier == turboQuantSupportUnsupported {
			support.UnsupportedReason = "hybrid KV architecture requires wrapper/reference handling"
		}
	}

	return support
}

func detectHeadDim(cfg fs.Config) (int, turboQuantHeadDimSource) {
	arch := strings.TrimSpace(cfg.Architecture())
	if value := readConfigUint(cfg, "attention.key_length", arch+".attention.key_length"); value > 0 {
		return int(value), turboQuantHeadDimSourceAttentionKeyLength
	}
	if value := readConfigUint(cfg, "attention.value_length", arch+".attention.value_length"); value > 0 {
		return int(value), turboQuantHeadDimSourceAttentionValueLength
	}

	embeddingLength := readConfigUint(cfg, "embedding_length", arch+".embedding_length")
	if headCountKV := readConfigUint(cfg, "attention.head_count_kv", arch+".attention.head_count_kv"); dividesCleanly(embeddingLength, headCountKV) {
		return int(embeddingLength / headCountKV), turboQuantHeadDimSourceEmbeddingHeadCountKV
	}
	if headCount := readConfigUint(cfg, "attention.head_count", arch+".attention.head_count"); dividesCleanly(embeddingLength, headCount) {
		return int(embeddingLength / headCount), turboQuantHeadDimSourceEmbeddingHeadCount
	}
	if value, ok := deriveUniformLayerHeadDim(cfg, embeddingLength, arch); ok {
		return value, turboQuantHeadDimSourceLayerHeadCountArray
	}
	if value := readConfigUint(cfg, "rope.dimension_count", arch+".rope.dimension_count"); value > 0 {
		return int(value), turboQuantHeadDimSourceRopeDimensionCount
	}
	return 0, turboQuantHeadDimSourceUnknown
}

func normalizeArchitectureClass(rawArch string, cfg fs.Config) string {
	arch := strings.ToLower(strings.TrimSpace(rawArch))
	switch arch {
	case "llama", "llama2", "llama3", "mixtral":
		return "llama-family"
	case "mistral":
		return "mistral-family"
	case "qwen2", "qwen3", "qwen2moe", "qwen3moe":
		return "qwen-family"
	case "gemma", "gemma2", "gemma3":
		return "gemma-family"
	case "phi", "phi2", "phi3", "phi3mini":
		return "phi-family"
	case "gptoss", "gpt-oss":
		return "gptoss-hybrid"
	case "deepseek2", "deepseekocr":
		return "deepseek-mla"
	case "glm4moelite":
		return "glm-mla"
	case "bert", "bert_embed":
		return "bert-like"
	}
	if hasMixedSlidingWindow(cfg) {
		return "sliding-window-hybrid"
	}
	if strings.Contains(arch, "vision") {
		return "vision-only"
	}
	return "unknown"
}

func detectHybridKVArchitecture(m model.Model, cfg fs.Config) bool {
	if _, ok := m.Config().Cache.(*kvcache.WrapperCache); ok {
		return true
	}
	if hasMixedSlidingWindow(cfg) {
		return true
	}
	headCounts := readConfigIntSlice(cfg, "attention.head_count")
	if hasVaryingPositiveValues(headCounts) {
		return true
	}
	headCountsKV := readConfigIntSlice(cfg, "attention.head_count_kv")
	if hasVaryingPositiveValues(headCountsKV) {
		return true
	}
	return false
}

func hasMixedSlidingWindow(cfg fs.Config) bool {
	values := readConfigIntSlice(cfg, "attention.sliding_window")
	if len(values) <= 1 {
		return false
	}
	seen := make(map[int]struct{})
	for _, value := range values {
		if value < 0 {
			continue
		}
		seen[value] = struct{}{}
	}
	return len(seen) > 1
}

func deriveUniformLayerHeadDim(cfg fs.Config, embeddingLength uint32, arch string) (int, bool) {
	if embeddingLength == 0 {
		return 0, false
	}
	for _, key := range []string{"attention.head_count", arch + ".attention.head_count", "attention.head_count_kv", arch + ".attention.head_count_kv"} {
		values := readConfigIntSlice(cfg, key)
		if len(values) == 0 {
			continue
		}
		if !allPositiveEqual(values) {
			return 0, false
		}
		headCount := uint32(values[0])
		if dividesCleanly(embeddingLength, headCount) {
			return int(embeddingLength / headCount), true
		}
	}
	return 0, false
}

func readConfigUint(cfg fs.Config, keys ...string) uint32 {
	for _, key := range keys {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		if value := cfg.Uint(key); value > 0 {
			return value
		}
		switch v := cfg.Value(key).(type) {
		case uint32:
			if v > 0 {
				return v
			}
		case int32:
			if v > 0 {
				return uint32(v)
			}
		case int:
			if v > 0 {
				return uint32(v)
			}
		case uint64:
			if v > 0 {
				return uint32(v)
			}
		}
	}
	return 0
}

func readConfigIntSlice(cfg fs.Config, key string) []int {
	if values := cfg.Ints(key); len(values) > 0 {
		out := make([]int, 0, len(values))
		for _, value := range values {
			out = append(out, int(value))
		}
		return out
	}
	raw := cfg.Value(key)
	switch values := raw.(type) {
	case []int:
		return append([]int(nil), values...)
	case []int32:
		out := make([]int, 0, len(values))
		for _, value := range values {
			out = append(out, int(value))
		}
		return out
	case []uint32:
		out := make([]int, 0, len(values))
		for _, value := range values {
			out = append(out, int(value))
		}
		return out
	}
	value := reflect.ValueOf(raw)
	if !value.IsValid() {
		return nil
	}
	if value.Kind() == reflect.Pointer {
		value = value.Elem()
	}
	if value.IsValid() && value.Kind() == reflect.Slice {
		out := make([]int, 0, value.Len())
		for i := 0; i < value.Len(); i++ {
			elem := value.Index(i)
			if !elem.IsValid() {
				continue
			}
			switch elem.Kind() {
			case reflect.Int, reflect.Int32, reflect.Int64:
				out = append(out, int(elem.Int()))
			case reflect.Uint, reflect.Uint32, reflect.Uint64:
				out = append(out, int(elem.Uint()))
			}
		}
		return out
	}
	return nil
}

func dividesCleanly(numerator, divisor uint32) bool {
	return numerator > 0 && divisor > 0 && numerator%divisor == 0 && numerator/divisor > 0
}

func allPositiveEqual(values []int) bool {
	if len(values) == 0 {
		return false
	}
	first := values[0]
	if first <= 0 {
		return false
	}
	for _, value := range values[1:] {
		if value != first || value <= 0 {
			return false
		}
	}
	return true
}

func hasVaryingPositiveValues(values []int) bool {
	if len(values) <= 1 {
		return false
	}
	seen := make(map[int]struct{})
	for _, value := range values {
		if value <= 0 {
			continue
		}
		seen[value] = struct{}{}
	}
	return len(seen) > 1
}
