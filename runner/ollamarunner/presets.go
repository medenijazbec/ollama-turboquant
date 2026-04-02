package ollamarunner

import "strings"

type turboQuantRolloutPreset string

const (
	turboQuantPresetSafe         turboQuantRolloutPreset = "safe"
	turboQuantPresetConservative turboQuantRolloutPreset = "conservative"
	turboQuantPresetExperimental turboQuantRolloutPreset = "experimental"
)

type turboQuantRecommendation struct {
	Preset           turboQuantRolloutPreset
	RecommendedK     string
	RecommendedV     string
	PairingValidated bool
	ExperimentalLane bool
	Warning          string
}

func normalizeTurboQuantRolloutPreset(value string) turboQuantRolloutPreset {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case string(turboQuantPresetSafe):
		return turboQuantPresetSafe
	case string(turboQuantPresetConservative):
		return turboQuantPresetConservative
	case string(turboQuantPresetExperimental):
		return turboQuantPresetExperimental
	default:
		return ""
	}
}

func inferTurboQuantRolloutPreset(support turboQuantModelSupport) turboQuantRolloutPreset {
	switch support.SupportTier {
	case turboQuantSupportSafe:
		return turboQuantPresetSafe
	case turboQuantSupportConservative:
		return turboQuantPresetConservative
	case turboQuantSupportExperimental:
		return turboQuantPresetExperimental
	default:
		return ""
	}
}

func resolveTurboQuantPreset(requestedPreset string, support turboQuantModelSupport, requestedK, requestedV, effectiveK, effectiveV, pathKind, fallbackReason string, faEnabled, vTurboSupported bool) turboQuantRecommendation {
	preset := normalizeTurboQuantRolloutPreset(requestedPreset)
	if preset == "" {
		preset = inferTurboQuantRolloutPreset(support)
	}

	rec := turboQuantRecommendation{Preset: preset}
	switch preset {
	case turboQuantPresetSafe:
		rec.RecommendedK = "tq35"
		rec.RecommendedV = "tq35"
	case turboQuantPresetConservative:
		if support.SupportTier == turboQuantSupportConservative {
			rec.RecommendedK = "q8_0"
			rec.RecommendedV = "tq35"
		} else {
			rec.RecommendedK = "tq35"
			rec.RecommendedV = "tq35"
		}
	case turboQuantPresetExperimental:
		rec.RecommendedK = effectiveK
		rec.RecommendedV = effectiveV
	}

	rec.PairingValidated = isValidatedTurboQuantPairing(support, effectiveK, effectiveV, faEnabled, vTurboSupported)
	rec.ExperimentalLane = preset == turboQuantPresetExperimental || isExperimentalTurboQuantPairing(requestedK, requestedV)

	warnings := make([]string, 0, 4)
	if requestedPreset != "" && preset != "" && !presetAllowedForSupportTier(preset, support.SupportTier) {
		warnings = append(warnings, "warning: preset "+string(preset)+" requested on support_tier="+string(support.SupportTier)+"; recommendation remains available but production-safe guarantees do not apply")
	}
	if support.ArchitectureClass == "gptoss-hybrid" || support.ArchitectureClass == "sliding-window-hybrid" || support.ArchitectureClass == "deepseek-mla" || support.ArchitectureClass == "glm-mla" || support.ArchitectureClass == "unknown" {
		if support.UnsupportedReason != "" {
			warnings = append(warnings, "warning: "+support.UnsupportedReason)
		}
	}
	if isExperimentalTurboQuantPairing(requestedK, requestedV) {
		warnings = append(warnings, "warning: requested TurboQuant pairing is outside the current validated recommendation set; treating it as experimental")
	}
	if preset == turboQuantPresetConservative && isTurboQuantKVType(requestedV) && requestedK != "q8_0" && requestedK != requestedV {
		warnings = append(warnings, "warning: q8_0-K + tq35-V is the only asymmetric conservative pairing currently endorsed in this branch")
	}
	if pathKind == "reference_wrapper" && strings.Contains(strings.ToLower(fallbackReason), "native turboquant disabled") {
		warnings = append(warnings, "warning: reference_wrapper path does not by itself prove backend-native KV memory savings")
	}
	if !rec.PairingValidated && (isTurboQuantKVType(effectiveK) || isTurboQuantKVType(effectiveV)) && preset != "" {
		warnings = append(warnings, "warning: requested TurboQuant pairing is outside the current validated recommendation set; treating it as experimental")
		rec.ExperimentalLane = true
	}

	rec.Warning = strings.Join(dedupeStrings(warnings), "; ")
	return rec
}

func presetAllowedForSupportTier(preset turboQuantRolloutPreset, tier turboQuantSupportTier) bool {
	switch preset {
	case turboQuantPresetSafe:
		return tier == turboQuantSupportSafe
	case turboQuantPresetConservative:
		return tier == turboQuantSupportSafe || tier == turboQuantSupportConservative
	case turboQuantPresetExperimental:
		return tier == turboQuantSupportSafe || tier == turboQuantSupportConservative || tier == turboQuantSupportExperimental
	default:
		return false
	}
}

func isValidatedTurboQuantPairing(support turboQuantModelSupport, effectiveK, effectiveV string, faEnabled, vTurboSupported bool) bool {
	if !isValidatedTurboQuantArchitecture(support.ArchitectureClass) {
		return false
	}
	switch {
	case support.SupportTier == turboQuantSupportSafe && effectiveK == "tq35" && effectiveV == "tq35":
		return true
	case (support.SupportTier == turboQuantSupportSafe || support.SupportTier == turboQuantSupportConservative) &&
		effectiveK == "q8_0" && effectiveV == "tq35" && vTurboSupported && faEnabled:
		return true
	default:
		return false
	}
}

func isExperimentalTurboQuantPairing(requestedK, requestedV string) bool {
	return requestedK == "tq25" || requestedV == "tq25"
}

func isValidatedTurboQuantArchitecture(architectureClass string) bool {
	switch architectureClass {
	case "llama-family", "mistral-family", "qwen-family", "gemma-family", "phi-family":
		return true
	default:
		return false
	}
}

func dedupeStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
