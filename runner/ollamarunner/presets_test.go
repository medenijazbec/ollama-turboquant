package ollamarunner

import "testing"

func TestResolveTurboQuantPresetSafePairing(t *testing.T) {
	rec := resolveTurboQuantPreset("safe", turboQuantModelSupport{
		ArchitectureClass: "llama-family",
		SupportTier:       turboQuantSupportSafe,
	}, "tq35", "tq35", "tq35", "tq35", "reference_wrapper", "", true, true)

	if rec.Preset != turboQuantPresetSafe {
		t.Fatalf("Preset = %q, want safe", rec.Preset)
	}
	if !rec.PairingValidated || rec.ExperimentalLane {
		t.Fatalf("unexpected recommendation flags: %+v", rec)
	}
	if rec.Warning != "" {
		t.Fatalf("unexpected warning: %q", rec.Warning)
	}
}

func TestResolveTurboQuantPresetConservativeValidatedAsymmetry(t *testing.T) {
	rec := resolveTurboQuantPreset("conservative", turboQuantModelSupport{
		ArchitectureClass: "qwen-family",
		SupportTier:       turboQuantSupportConservative,
	}, "q8_0", "tq35", "q8_0", "tq35", "reference_wrapper", "", true, true)

	if rec.Preset != turboQuantPresetConservative {
		t.Fatalf("Preset = %q, want conservative", rec.Preset)
	}
	if !rec.PairingValidated || rec.ExperimentalLane {
		t.Fatalf("unexpected recommendation flags: %+v", rec)
	}
	if rec.RecommendedK != "q8_0" || rec.RecommendedV != "tq35" {
		t.Fatalf("unexpected recommendation pairing: %+v", rec)
	}
}

func TestResolveTurboQuantPresetWarnsOnUnvalidatedAsymmetry(t *testing.T) {
	rec := resolveTurboQuantPreset("conservative", turboQuantModelSupport{
		ArchitectureClass: "llama-family",
		SupportTier:       turboQuantSupportConservative,
	}, "q4_0", "tq35", "q4_0", "tq35", "reference_wrapper", "", true, true)

	if rec.PairingValidated {
		t.Fatalf("PairingValidated = true, want false: %+v", rec)
	}
	if !rec.ExperimentalLane {
		t.Fatalf("ExperimentalLane = false, want true: %+v", rec)
	}
	if rec.Warning == "" {
		t.Fatal("expected warning for unvalidated asymmetry")
	}
}

func TestResolveTurboQuantPresetMarksExperimentalPairing(t *testing.T) {
	rec := resolveTurboQuantPreset("experimental", turboQuantModelSupport{
		ArchitectureClass: "llama-family",
		SupportTier:       turboQuantSupportExperimental,
	}, "tq25", "tq25", "tq25", "tq25", "reference_wrapper", "", true, true)

	if rec.Preset != turboQuantPresetExperimental {
		t.Fatalf("Preset = %q, want experimental", rec.Preset)
	}
	if !rec.ExperimentalLane {
		t.Fatalf("ExperimentalLane = false, want true: %+v", rec)
	}
	if rec.PairingValidated {
		t.Fatalf("PairingValidated = true, want false: %+v", rec)
	}
	if rec.Warning == "" {
		t.Fatal("expected warning for experimental pairing")
	}
}

func TestResolveTurboQuantPresetWarnsOnProvisionalArchitecture(t *testing.T) {
	rec := resolveTurboQuantPreset("safe", turboQuantModelSupport{
		ArchitectureClass: "deepseek-mla",
		SupportTier:       turboQuantSupportUnsupported,
		UnsupportedReason: "MLA / compressed-KV topology is provisional and not yet supported for native TurboQuant activation",
	}, "tq35", "tq35", "tq35", "tq35", "reference_wrapper", "", true, true)

	if rec.Warning == "" {
		t.Fatal("expected warning for provisional architecture")
	}
	if rec.PairingValidated {
		t.Fatalf("PairingValidated = true, want false: %+v", rec)
	}
}
