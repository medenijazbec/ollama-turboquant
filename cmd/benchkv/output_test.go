package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteCSV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.csv")
	rows := []epochAggregate{{
		Host: "http://127.0.0.1:11438", HostLabel: "baseline", ServerVersion: "0.18.3", Model: "m", KVModeRequested: "f16", KVModeResolved: "f16", KVAlgoResolved: "", KVPath: "dense-fallback",
		Workload: "prefill-heavy", NumCtx: 8192, PromptTokensTarget: 7372, MaxTokens: 128, Concurrency: 1, Epoch: 1, Status: statusOK, Success: boolPtr(true),
	}}
	if err := writeCSV(path, rows); err != nil {
		t.Fatalf("writeCSV failed: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if !strings.Contains(string(data), "host,host_label,server_version,model") {
		t.Fatal("expected csv header")
	}
	if !strings.Contains(string(data), "quant,kv_mode_requested,kv_mode_requested_k,kv_mode_requested_v,kv_mode_resolved") {
		t.Fatal("expected extended csv header")
	}
	if !strings.Contains(string(data), "requested_mode,effective_mode") {
		t.Fatal("expected requested/effective mode csv header")
	}
	if !strings.Contains(string(data), "fallback_applied,k_only_fallback") {
		t.Fatal("expected fallback columns in csv header")
	}
	if !strings.Contains(string(data), "fa_required_for_v_turbo") {
		t.Fatal("expected Flash Attention gating column in csv header")
	}
	if !strings.Contains(string(data), "detected_head_dim,architecture_class,support_tier,hybrid_kv_architecture") {
		t.Fatal("expected support-matrix columns in csv header")
	}
	if !strings.Contains(string(data), "gpu_stats_source,host_stats_source,validation_kind,validation_status") {
		t.Fatal("expected telemetry/validation columns in csv header")
	}
	if !strings.Contains(string(data), "requested_num_ctx,attempted_num_ctx,effective_num_ctx") {
		t.Fatal("expected large-context columns in csv header")
	}
	if !strings.Contains(string(data), "visible_gpu_count,per_gpu_vram_gib,total_visible_vram_bytes,process_vram_bytes") {
		t.Fatal("expected per-gpu telemetry columns in csv header")
	}
	if !strings.Contains(string(data), "peak_host_ram_delta_bytes,used_host_assist,used_mmap,used_cpu_assist") {
		t.Fatal("expected host-assist columns in csv header")
	}
	if !strings.Contains(string(data), "kv_algo_resolved") {
		t.Fatal("expected kv algorithm csv header")
	}
}

func TestValidateResolvedKVModesRequiresExplicitFallbackMetadata(t *testing.T) {
	row := workerResult{
		KVModeRequested:     "tq35",
		KVModeRequestedK:    "tq35",
		KVModeRequestedV:    "tq35",
		KVModeResolved:      "mixed",
		KVModeResolvedK:     "tq35",
		KVModeResolvedV:     "f16",
		RequestedMode:       "tq35",
		EffectiveMode:       "k=tq35,v=f16",
		KVAlgoResolved:      "paper",
		KVPath:              "mixed",
		KVPathK:             "turboquant-cpu-fastpath",
		KVPathV:             "dense-fallback",
		FallbackApplied:     false,
		KOnlyFallback:       false,
		FARequiredForVTurbo: true,
		FAEnabled:           false,
		VTurboSupported:     false,
	}

	if err := validateResolvedKVModes(row); err == nil {
		t.Fatal("expected ambiguous downgrade to fail validation")
	}
}

func TestValidateResolvedKVModesAllowsExplicitKOnlyFallback(t *testing.T) {
	row := workerResult{
		KVModeRequested:     "tq35",
		KVModeRequestedK:    "tq35",
		KVModeRequestedV:    "tq35",
		KVModeResolved:      "mixed",
		KVModeResolvedK:     "tq35",
		KVModeResolvedV:     "f16",
		RequestedMode:       "tq35",
		EffectiveMode:       "k=tq35,v=f16",
		KVAlgoResolved:      "paper",
		KVPath:              "mixed",
		KVPathK:             "turboquant-cpu-fastpath",
		KVPathV:             "dense-fallback",
		FallbackApplied:     true,
		KOnlyFallback:       true,
		FallbackReason:      "requested V turboquant requires Flash Attention on the active backend; falling back to f16 on V",
		FARequiredForVTurbo: true,
		FAEnabled:           false,
		VTurboSupported:     false,
	}

	if err := validateResolvedKVModes(row); err != nil {
		t.Fatalf("expected explicit K-only fallback to validate, got %v", err)
	}
}

func TestWriteSummaryExcludesFallbackAndFailedValidationRows(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "summary.md")
	rows := []epochAggregate{
		{
			HostLabel:        "turbo",
			KVModeRequested:  "tq35",
			EffectiveMode:    "tq35",
			Workload:         string(workloadDecodeGrowth),
			NumCtx:           8192,
			Concurrency:      1,
			Status:           statusOK,
			FullGPUResidency: true,
			ValidationStatus: string(validationFailed),
		},
		{
			HostLabel:        "turbo",
			KVModeRequested:  "tq35",
			EffectiveMode:    "tq35",
			Workload:         string(workloadDecodeGrowth),
			NumCtx:           8192,
			Concurrency:      1,
			Status:           statusOK,
			FullGPUResidency: true,
			FallbackApplied:  true,
			ValidationStatus: string(validationPassed),
		},
	}
	if err := writeSummary(path, rows, nil); err != nil {
		t.Fatalf("writeSummary failed: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if strings.Contains(string(data), "claim: On the TurboQuant runtime") {
		t.Fatal("expected summary to exclude fallback or failed-validation claims")
	}
}

func TestWriteSummaryIncludesLargeContextSection(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "summary-large-context.md")
	rows := []epochAggregate{
		{
			HostLabel:        "turbo",
			KVModeRequested:  "q8_0/tq35",
			RequestedMode:    "k=q8_0,v=tq35",
			EffectiveMode:    "k=q8_0,v=tq35",
			Workload:         string(workloadLongContextRecall),
			NumCtx:           262144,
			RequestedNumCtx:  1000000,
			EffectiveNumCtx:  262144,
			Concurrency:      1,
			Status:           statusOK,
			FullGPUResidency: true,
			ValidationStatus: string(validationPassed),
		},
	}
	if err := writeSummary(path, rows, nil); err != nil {
		t.Fatalf("writeSummary failed: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if !strings.Contains(string(data), "# Large-Context Summary") {
		t.Fatal("expected large-context summary section")
	}
	if !strings.Contains(string(data), "ctx=1000000->262144") {
		t.Fatal("expected requested->effective context summary")
	}
}
