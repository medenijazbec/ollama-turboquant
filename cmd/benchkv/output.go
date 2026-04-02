package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
)

func writeJSONL(path string, rows []workerResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

func writeCSV(path string, rows []epochAggregate) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{
		"host", "host_label", "server_version", "model", "quant", "kv_mode_requested", "kv_mode_requested_k", "kv_mode_requested_v", "kv_mode_resolved", "kv_mode_resolved_k", "kv_mode_resolved_v", "requested_mode", "effective_mode", "kv_algo_resolved", "kv_algo_resolved_k", "kv_algo_resolved_v", "kv_path", "kv_path_k", "kv_path_v", "kv_symmetric", "kv_asymmetric", "fallback_applied", "k_only_fallback", "fallback_reason", "turboquant_path_kind", "native_turboquant_active", "reference_turboquant_active", "fa_enabled", "fa_required_for_v_turbo", "v_turbo_supported", "tq_block_size",
		"workload", "num_ctx", "concurrency", "prompt_tokens_target", "prompt_eval_count", "max_tokens", "eval_count",
		"generated_tokens", "live_kv_tokens_total", "ctx_x_conc", "epoch", "warmup", "prefill_tps", "decode_tps",
		"ttft_ms_mean", "ttft_ms_p95", "load_ms", "total_ms", "wall_ms", "peak_vram_bytes", "avg_gpu_util", "peak_gpu_util",
		"gpu_metrics_available", "host_ram_used_bytes", "peak_host_ram_bytes", "host_metrics_available",
		"processor_state_before", "processor_state_after", "full_gpu_residency", "spilled", "gpu_offload_regression",
		"runner_rss_bytes", "success", "status", "error",
	}
	if err := w.Write(header); err != nil {
		return err
	}

	for _, row := range rows {
		record := []string{
			row.Host,
			row.HostLabel,
			row.ServerVersion,
			row.Model,
			row.Quant,
			row.KVModeRequested,
			row.KVModeRequestedK,
			row.KVModeRequestedV,
			row.KVModeResolved,
			row.KVModeResolvedK,
			row.KVModeResolvedV,
			row.RequestedMode,
			row.EffectiveMode,
			row.KVAlgoResolved,
			row.KVAlgoResolvedK,
			row.KVAlgoResolvedV,
			row.KVPath,
			row.KVPathK,
			row.KVPathV,
			fmt.Sprintf("%t", row.KVSymmetric),
			fmt.Sprintf("%t", row.KVAsymmetric),
			fmt.Sprintf("%t", row.FallbackApplied),
			fmt.Sprintf("%t", row.KOnlyFallback),
			row.FallbackReason,
			row.TurboQuantPathKind,
			fmt.Sprintf("%t", row.NativeTurboQuantActive),
			fmt.Sprintf("%t", row.ReferenceTurboQuantActive),
			fmt.Sprintf("%t", row.FAEnabled),
			fmt.Sprintf("%t", row.FARequiredForVTurbo),
			fmt.Sprintf("%t", row.VTurboSupported),
			fmt.Sprintf("%d", row.TQBlockSize),
			row.Workload,
			fmt.Sprintf("%d", row.NumCtx),
			fmt.Sprintf("%d", row.Concurrency),
			fmt.Sprintf("%d", row.PromptTokensTarget),
			fmt.Sprintf("%d", row.PromptEvalCount),
			fmt.Sprintf("%d", row.MaxTokens),
			fmt.Sprintf("%d", row.EvalCount),
			fmt.Sprintf("%d", row.GeneratedTokens),
			fmt.Sprintf("%d", row.LiveKVTokensTotal),
			fmt.Sprintf("%d", row.CtxXConc),
			fmt.Sprintf("%d", row.Epoch),
			fmt.Sprintf("%t", row.Warmup),
			formatPerfFloatCSV(row.Status, row.PrefillTPS),
			formatPerfFloatCSV(row.Status, row.DecodeTPS),
			formatPerfFloatCSV(row.Status, row.TTFTMSMean),
			formatPerfFloatCSV(row.Status, row.TTFTMSP95),
			formatPerfFloatCSV(row.Status, row.LoadMS),
			formatPerfFloatCSV(row.Status, row.TotalMS),
			formatPerfFloatCSV(row.Status, row.WallMS),
			formatInt64CSV(row.PeakVRAMBytes),
			formatFloat64CSV(row.AvgGPUUtil),
			formatFloat64CSV(row.PeakGPUUtil),
			fmt.Sprintf("%t", row.GPUMetricsAvailable),
			formatInt64CSV(row.HostRAMUsedBytes),
			formatInt64CSV(row.PeakHostRAMBytes),
			fmt.Sprintf("%t", row.HostMetricsAvailable),
			row.ProcessorStateBefore,
			row.ProcessorStateAfter,
			fmt.Sprintf("%t", row.FullGPUResidency),
			fmt.Sprintf("%t", row.Spilled),
			fmt.Sprintf("%t", row.GPUOffloadRegression),
			fmt.Sprintf("%d", row.RunnerRSSBytes),
			formatBoolCSV(row.Success),
			string(row.Status),
			row.Error,
		}
		if err := w.Write(record); err != nil {
			return err
		}
	}
	return w.Error()
}

func writeSummary(path string, rows []epochAggregate, staircases []staircaseRecord) error {
	var b strings.Builder
	sameRuntimeHosts := sameRuntimeHostSet(rows)

	b.WriteString("# Regression Summary\n\n")
	writeSupportCounts(&b, rows, func(row epochAggregate) bool {
		return row.KVModeRequested == "f16" && row.Workload == string(workloadDecodeGrowth)
	})
	writeRegressionSummary(&b, rows)

	b.WriteString("\n# TurboQuant Same-Runtime Summary\n\n")
	writeSupportCounts(&b, rows, func(row epochAggregate) bool {
		if _, ok := sameRuntimeHosts[row.HostLabel]; !ok {
			return false
		}
		return row.EffectiveMode == "f16" || strings.HasPrefix(row.EffectiveMode, "tq")
	})
	writeSameRuntimeSummary(&b, rows, sameRuntimeHosts)

	b.WriteString("\n# Product Claim vs Baseline\n\n")
	writeSupportCounts(&b, rows, func(row epochAggregate) bool {
		return (row.HostLabel == "baseline" && row.EffectiveMode == "f16") || (row.HostLabel == "turbo" && row.EffectiveMode == "tq35")
	})
	writeProductSummary(&b, rows)

	b.WriteString("\n# Memory / Capacity Summary\n\n")
	writeSupportCounts(&b, rows, func(row epochAggregate) bool {
		return row.Workload == string(workloadNearOOMStaircase)
	})
	writeCapacitySummary(&b, staircases)

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func writeSupportCounts(b *strings.Builder, rows []epochAggregate, include func(epochAggregate) bool) {
	supported := 0
	unsupported := 0
	failed := 0
	for _, row := range rows {
		if row.Warmup || !include(row) {
			continue
		}
		switch row.Status {
		case statusOK:
			supported++
		case statusUnsupported:
			unsupported++
		case statusFailed:
			failed++
		}
	}
	b.WriteString(fmt.Sprintf("- supported=%d unsupported=%d failed=%d\n", supported, unsupported, failed))
}

func writeRegressionSummary(b *strings.Builder, rows []epochAggregate) {
	grouped := summarizeRows(rows, func(row epochAggregate) bool {
		return !row.Warmup && row.Status == statusOK && row.KVModeRequested == "f16" && row.Workload == string(workloadDecodeGrowth)
	})
	if len(grouped) == 0 {
		b.WriteString("- no regression rows available\n")
		return
	}

	for _, group := range grouped {
		b.WriteString(fmt.Sprintf(
			"- %s | f16 | %s | ctx=%d | conc=%d | prefill=%.2f tok/s | decode=%.2f tok/s | ttft=%.2f ms | total=%.2f ms\n",
			group.HostLabel, group.Workload, group.NumCtx, group.Concurrency,
			group.PrefillTPS, group.DecodeTPS, group.TTFTMSMean, group.TotalMS,
		))
	}

	for _, pair := range pairedGroups(grouped, func(group summaryGroup) bool { return group.EffectiveMode == "f16" }) {
		if pair.Baseline.HostLabel == "" || pair.Turbo.HostLabel == "" {
			continue
		}
		b.WriteString(fmt.Sprintf(
			"- claim: On the same model and workload, the TurboQuant fork in f16 is within %.2f%% of stock f16 for prefill, %.2f%% for decode, %.2f%% for TTFT, and %.2f%% for total time at ctx=%d conc=%d\n",
			abs(deltaPercent(pair.Baseline.PrefillTPS, pair.Turbo.PrefillTPS)),
			abs(deltaPercent(pair.Baseline.DecodeTPS, pair.Turbo.DecodeTPS)),
			abs(deltaPercent(pair.Baseline.TTFTMSMean, pair.Turbo.TTFTMSMean)),
			abs(deltaPercent(pair.Baseline.TotalMS, pair.Turbo.TotalMS)),
			pair.Baseline.NumCtx,
			pair.Baseline.Concurrency,
		))
	}
}

func writeSameRuntimeSummary(b *strings.Builder, rows []epochAggregate, sameRuntimeHosts map[string]struct{}) {
	grouped := summarizeRows(rows, func(row epochAggregate) bool {
		if _, ok := sameRuntimeHosts[row.HostLabel]; !ok {
			return false
		}
		if row.FallbackApplied {
			return false
		}
		return !row.Warmup && row.Status == statusOK && row.FullGPUResidency && !row.Spilled && (row.EffectiveMode == "f16" || strings.HasPrefix(row.EffectiveMode, "tq"))
	})
	if len(grouped) == 0 {
		b.WriteString("- no same-runtime full-GPU rows available\n")
		return
	}

	for _, key := range groupedKeys(grouped) {
		base, ok := findGroup(grouped, key, "f16")
		if !ok {
			continue
		}
		b.WriteString(fmt.Sprintf(
			"- %s | f16 | %s | ctx=%d | conc=%d | prefill=%.2f tok/s | decode=%.2f tok/s | ttft=%.2f ms | live_kv=%d | peak_vram=%s\n",
			key.HostLabel, key.Workload, key.NumCtx, key.Concurrency,
			base.PrefillTPS, base.DecodeTPS, base.TTFTMSMean, base.LiveKVTokensTotal, formatInt64Summary(base.PeakVRAMBytes),
		))
		for _, kvMode := range []string{"tq25", "tq35"} {
			group, exists := findGroup(grouped, key, kvMode)
			if !exists {
				continue
			}
			b.WriteString(fmt.Sprintf(
				"- claim: On the TurboQuant runtime, %s vs f16 improves prefill by %.2f%%, decode by %.2f%%, reduces TTFT by %.2f%%, and changes live KV occupancy by %.2f%% at ctx=%d conc=%d\n",
				kvMode,
				deltaPercent(base.PrefillTPS, group.PrefillTPS),
				deltaPercent(base.DecodeTPS, group.DecodeTPS),
				deltaPercent(base.TTFTMSMean, group.TTFTMSMean)*-1,
				deltaPercentFloat64(float64(base.LiveKVTokensTotal), float64(group.LiveKVTokensTotal)),
				key.NumCtx,
				key.Concurrency,
			))
		}
	}
}

func writeProductSummary(b *strings.Builder, rows []epochAggregate) {
	grouped := summarizeRows(rows, func(row epochAggregate) bool {
		if row.Warmup || row.Status != statusOK || !row.FullGPUResidency || row.Spilled || row.FallbackApplied {
			return false
		}
		return (row.HostLabel == "baseline" && row.EffectiveMode == "f16") || (row.HostLabel == "turbo" && row.EffectiveMode == "tq35")
	})
	if len(grouped) == 0 {
		b.WriteString("- no baseline-vs-fork product rows available\n")
		return
	}

	for _, pair := range pairedGroups(grouped, func(group summaryGroup) bool {
		return (group.HostLabel == "baseline" && group.EffectiveMode == "f16") || (group.HostLabel == "turbo" && group.EffectiveMode == "tq35")
	}) {
		if pair.Baseline.HostLabel == "" || pair.Turbo.HostLabel == "" {
			continue
		}
		b.WriteString(fmt.Sprintf(
			"- product: At %s ctx=%d conc=%d, turbo tq35 vs stock f16 delivers %.2f%% prefill, %.2f%% decode, and %.2f%% more live KV tokens on the same hardware\n",
			pair.Baseline.Workload,
			pair.Baseline.NumCtx,
			pair.Baseline.Concurrency,
			deltaPercent(pair.Baseline.PrefillTPS, pair.Turbo.PrefillTPS),
			deltaPercent(pair.Baseline.DecodeTPS, pair.Turbo.DecodeTPS),
			deltaPercentFloat64(float64(pair.Baseline.LiveKVTokensTotal), float64(pair.Turbo.LiveKVTokensTotal)),
		))
	}
}

func writeCapacitySummary(b *strings.Builder, staircases []staircaseRecord) {
	if len(staircases) == 0 {
		b.WriteString("- no capacity rows available\n")
		return
	}

	slices.SortFunc(staircases, func(a, c staircaseRecord) int {
		if a.HostLabel != c.HostLabel {
			return strings.Compare(a.HostLabel, c.HostLabel)
		}
		return strings.Compare(a.KVMode, c.KVMode)
	})

	for _, record := range staircases {
		b.WriteString(fmt.Sprintf(
			"- %s | %s | max_full_gpu_ctx=%d | max_full_gpu_ctx_x_conc=%d | max_stable_ctx_x_conc=%d | max_live_kv_tokens_total=%d | first_spill_ctx_x_conc=%d | first_failure_ctx_x_conc=%d | reason=%s\n",
			record.HostLabel,
			record.KVMode,
			record.MaxFullGPUCtx,
			record.LastFullGPUCtxXConcurrency,
			record.MaxStableCtxXConcurrency,
			record.MaxLiveKVTokensTotal,
			record.FirstSpillCtxXConcurrency,
			record.FirstFailureCtxXConcurrency,
			record.FailureReason,
		))
	}

	byHost := make(map[string]map[string]staircaseRecord)
	for _, record := range staircases {
		if byHost[record.HostLabel] == nil {
			byHost[record.HostLabel] = make(map[string]staircaseRecord)
		}
		byHost[record.HostLabel][record.KVMode] = record
	}
	for hostLabel, modes := range byHost {
		base, ok := modes["f16"]
		if !ok {
			continue
		}
		for _, kvMode := range []string{"tq25", "tq35"} {
			record, exists := modes[kvMode]
			if !exists {
				continue
			}
			b.WriteString(fmt.Sprintf(
				"- claim: On %s, %s increases max full-GPU ctx x conc by %.2f%% and max stable ctx x conc by %.2f%% before failure\n",
				hostLabel,
				kvMode,
				deltaPercentFloat64(float64(base.LastFullGPUCtxXConcurrency), float64(record.LastFullGPUCtxXConcurrency)),
				deltaPercentFloat64(float64(base.MaxStableCtxXConcurrency), float64(record.MaxStableCtxXConcurrency)),
			))
		}
	}
}

type summaryGroup struct {
	Host              string
	HostLabel         string
	ServerVersion     string
	Model             string
	Quant             string
	KVModeRequested   string
	EffectiveMode     string
	Workload          string
	NumCtx            int
	Concurrency       int
	PrefillTPS        float64
	DecodeTPS         float64
	TTFTMSMean        float64
	TotalMS           float64
	LiveKVTokensTotal int
	PeakVRAMBytes     *int64
	PeakHostRAMBytes  *int64
	SupportedCount    int
}

type summaryKey struct {
	HostLabel   string
	Workload    string
	NumCtx      int
	Concurrency int
}

type comparisonPair struct {
	Baseline summaryGroup
	Turbo    summaryGroup
}

func summarizeRows(rows []epochAggregate, include func(epochAggregate) bool) []summaryGroup {
	type key struct {
		host        string
		hostLabel   string
		version     string
		model       string
		quant       string
		kvMode      string
		effective   string
		workload    string
		numCtx      int
		concurrency int
	}
	type acc struct {
		count         int
		prefillTPS    float64
		decodeTPS     float64
		ttft          float64
		totalMS       float64
		liveKV        float64
		peakVRAMBytes *int64
		peakHostRAM   *int64
	}

	m := make(map[key]*acc)
	for _, row := range rows {
		if !include(row) {
			continue
		}
		k := key{row.Host, row.HostLabel, row.ServerVersion, row.Model, row.Quant, row.KVModeRequested, row.EffectiveMode, row.Workload, row.NumCtx, row.Concurrency}
		if m[k] == nil {
			m[k] = &acc{}
		}
		m[k].count++
		m[k].prefillTPS += row.PrefillTPS
		m[k].decodeTPS += row.DecodeTPS
		m[k].ttft += row.TTFTMSMean
		m[k].totalMS += row.TotalMS
		m[k].liveKV += float64(row.LiveKVTokensTotal)
		if ptrInt64Value(row.PeakVRAMBytes) > ptrInt64Value(m[k].peakVRAMBytes) {
			m[k].peakVRAMBytes = row.PeakVRAMBytes
		}
		if ptrInt64Value(row.PeakHostRAMBytes) > ptrInt64Value(m[k].peakHostRAM) {
			m[k].peakHostRAM = row.PeakHostRAMBytes
		}
	}

	out := make([]summaryGroup, 0, len(m))
	for k, v := range m {
		if v.count == 0 {
			continue
		}
		out = append(out, summaryGroup{
			Host:              k.host,
			HostLabel:         k.hostLabel,
			ServerVersion:     k.version,
			Model:             k.model,
			Quant:             k.quant,
			KVModeRequested:   k.kvMode,
			EffectiveMode:     firstNonEmpty(k.effective, k.kvMode),
			Workload:          k.workload,
			NumCtx:            k.numCtx,
			Concurrency:       k.concurrency,
			PrefillTPS:        v.prefillTPS / float64(v.count),
			DecodeTPS:         v.decodeTPS / float64(v.count),
			TTFTMSMean:        v.ttft / float64(v.count),
			TotalMS:           v.totalMS / float64(v.count),
			LiveKVTokensTotal: int(v.liveKV / float64(v.count)),
			PeakVRAMBytes:     v.peakVRAMBytes,
			PeakHostRAMBytes:  v.peakHostRAM,
			SupportedCount:    v.count,
		})
	}

	slices.SortFunc(out, func(a, b summaryGroup) int {
		if a.HostLabel != b.HostLabel {
			return strings.Compare(a.HostLabel, b.HostLabel)
		}
		if a.KVModeRequested != b.KVModeRequested {
			return strings.Compare(a.KVModeRequested, b.KVModeRequested)
		}
		if a.Workload != b.Workload {
			return strings.Compare(a.Workload, b.Workload)
		}
		if a.NumCtx != b.NumCtx {
			return a.NumCtx - b.NumCtx
		}
		return a.Concurrency - b.Concurrency
	})
	return out
}

func sameRuntimeHostSet(rows []epochAggregate) map[string]struct{} {
	out := make(map[string]struct{})
	for _, row := range rows {
		if row.Status != statusUnsupported && strings.HasPrefix(row.KVModeRequested, "tq") {
			out[row.HostLabel] = struct{}{}
		}
	}
	return out
}

func groupedKeys(groups []summaryGroup) []summaryKey {
	seen := make(map[summaryKey]struct{})
	var keys []summaryKey
	for _, group := range groups {
		key := summaryKey{HostLabel: group.HostLabel, Workload: group.Workload, NumCtx: group.NumCtx, Concurrency: group.Concurrency}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(a, b summaryKey) int {
		if a.HostLabel != b.HostLabel {
			return strings.Compare(a.HostLabel, b.HostLabel)
		}
		if a.Workload != b.Workload {
			return strings.Compare(a.Workload, b.Workload)
		}
		if a.NumCtx != b.NumCtx {
			return a.NumCtx - b.NumCtx
		}
		return a.Concurrency - b.Concurrency
	})
	return keys
}

func findGroup(groups []summaryGroup, key summaryKey, kvMode string) (summaryGroup, bool) {
	for _, group := range groups {
		if group.HostLabel == key.HostLabel && group.Workload == key.Workload && group.NumCtx == key.NumCtx && group.Concurrency == key.Concurrency && group.EffectiveMode == kvMode {
			return group, true
		}
	}
	return summaryGroup{}, false
}

func pairedGroups(groups []summaryGroup, include func(summaryGroup) bool) []comparisonPair {
	type pairKey struct {
		workload    string
		numCtx      int
		concurrency int
	}

	pairs := make(map[pairKey]*comparisonPair)
	for _, group := range groups {
		if !include(group) {
			continue
		}
		key := pairKey{workload: group.Workload, numCtx: group.NumCtx, concurrency: group.Concurrency}
		if pairs[key] == nil {
			pairs[key] = &comparisonPair{}
		}
		switch {
		case group.HostLabel == "baseline" && group.EffectiveMode == "f16":
			pairs[key].Baseline = group
		case group.HostLabel == "turbo" && (group.EffectiveMode == "f16" || group.EffectiveMode == "tq35"):
			pairs[key].Turbo = group
		}
	}

	keys := make([]pairKey, 0, len(pairs))
	for key := range pairs {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(a, b pairKey) int {
		if a.workload != b.workload {
			return strings.Compare(a.workload, b.workload)
		}
		if a.numCtx != b.numCtx {
			return a.numCtx - b.numCtx
		}
		return a.concurrency - b.concurrency
	})

	out := make([]comparisonPair, 0, len(keys))
	for _, key := range keys {
		out = append(out, *pairs[key])
	}
	return out
}

func deltaPercent(base, current float64) float64 {
	return deltaPercentFloat64(base, current)
}

func deltaPercentFloat64(base, current float64) float64 {
	if base == 0 {
		return 0
	}
	return ((current - base) / base) * 100
}

func abs(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

func formatInt64CSV(v *int64) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%d", *v)
}

func formatFloat64CSV(v *float64) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%.4f", *v)
}

func formatPerfFloatCSV(status resultStatus, v float64) string {
	if status == statusUnsupported {
		return ""
	}
	return fmt.Sprintf("%.4f", v)
}

func formatBoolCSV(v *bool) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%t", *v)
}

func formatInt64Summary(v *int64) string {
	if v == nil {
		return "n/a"
	}
	return fmt.Sprintf("%d", *v)
}
