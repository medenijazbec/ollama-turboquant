package main

import "testing"

func TestMakeWorkloadSpecDecodeGrowthSkipsInvalid(t *testing.T) {
	cfg := config{}
	if _, ok := makeWorkloadSpec(cfg, workloadDecodeGrowth, 2048, 1); ok {
		t.Fatal("expected invalid decode-growth spec for low context")
	}
}

func TestMakeWorkloadSpecPrefillHeavy(t *testing.T) {
	cfg := config{}
	spec, ok := makeWorkloadSpec(cfg, workloadPrefillHeavy, 8192, 2)
	if !ok {
		t.Fatal("expected valid prefill-heavy spec")
	}
	if spec.PromptTokensTarget != promptTokensAt90Percent(8192) {
		t.Fatalf("unexpected prompt token target: %d", spec.PromptTokensTarget)
	}
	if spec.MaxTokens != 128 {
		t.Fatalf("unexpected max tokens: %d", spec.MaxTokens)
	}
}

func TestBuildImpactCells(t *testing.T) {
	cfg := config{
		Profile: "impact",
		Hosts:   []hostTarget{{Label: "baseline"}},
		KVModes: []string{"f16"},
	}

	cells := buildStandardCells(cfg)
	if len(cells) != 2 {
		t.Fatalf("cell count = %d, want 2", len(cells))
	}

	prefill := cells[0].Workload
	if prefill.Name != workloadPrefillHeavy || prefill.NumCtx != 16384 || prefill.Concurrency != 2 || prefill.MaxTokens != 128 {
		t.Fatalf("unexpected impact prefill cell: %#v", prefill)
	}

	decode := cells[1].Workload
	if decode.Name != workloadDecodeGrowth || decode.NumCtx != 8192 || decode.Concurrency != 1 || decode.MaxTokens != 768 || decode.PromptTokensTarget != 2048 {
		t.Fatalf("unexpected impact decode cell: %#v", decode)
	}
}

func TestBuildTurboBenefitCellsUsesRequestHostsOnly(t *testing.T) {
	cfg := config{
		Profile: "turbo-benefit",
		Hosts: []hostTarget{
			{Label: "baseline", KVSupportMode: hostKVSupportLegacy},
			{Label: "turbo", KVSupportMode: hostKVSupportRequest},
		},
		KVModes:     []string{"f16", "tq35"},
		Workloads:   []workloadName{workloadPrefillHeavy, workloadDecodeGrowth},
		NumCtx:      []int{8192},
		Concurrency: []int{1},
	}

	cells := buildStandardCells(cfg)
	if len(cells) != 4 {
		t.Fatalf("cell count = %d, want 4", len(cells))
	}
	for _, cell := range cells {
		if cell.Host.Label != "turbo" {
			t.Fatalf("unexpected non-request host in turbo-benefit cells: %#v", cell)
		}
	}
}

func TestSpillProfileBuildsNoStandardCells(t *testing.T) {
	cfg := config{
		Profile: "spill",
		Hosts: []hostTarget{
			{Label: "baseline", KVSupportMode: hostKVSupportLegacy},
			{Label: "turbo", KVSupportMode: hostKVSupportRequest},
		},
		KVModes:     []string{"f16", "tq35"},
		NumCtx:      []int{8192, 16384},
		Concurrency: []int{1, 2, 4, 8},
	}

	cells := buildStandardCells(cfg)
	if len(cells) != 0 {
		t.Fatalf("spill profile should not build standard cells, got %d", len(cells))
	}
}

func TestMakeWorkloadSpecLargeContext(t *testing.T) {
	cfg := config{MinFitDecode: 32, StretchContext: 1000000}
	spec, ok := makeWorkloadSpec(cfg, workloadFitCeiling, 1000000, 1)
	if !ok {
		t.Fatal("expected valid fit-ceiling spec")
	}
	if spec.MaxTokens != 32 {
		t.Fatalf("fit-ceiling MaxTokens = %d, want 32", spec.MaxTokens)
	}
	spec, ok = makeWorkloadSpec(cfg, workloadDecodeCorruption, 262144, 1)
	if !ok || spec.MaxTokens != 2048 {
		t.Fatalf("unexpected decode-corruption spec: %#v", spec)
	}
}

func TestBuildLargeContextCells(t *testing.T) {
	cfg := config{
		Profile:        "large-context",
		Hosts:          []hostTarget{{Label: "turbo"}},
		KVModes:        []string{"f16", "q8_0/tq35", "tq35"},
		Workloads:      []workloadName{workloadFitCeiling, workloadLongContextRecall},
		StretchContext: 1000000,
		MinFitDecode:   32,
	}
	cells := buildStandardCells(cfg)
	if len(cells) != 6 {
		t.Fatalf("cell count = %d, want 6", len(cells))
	}
	if cells[0].Workload.NumCtx != 1000000 {
		t.Fatalf("first large-context cell num_ctx = %d, want 1000000", cells[0].Workload.NumCtx)
	}
}
