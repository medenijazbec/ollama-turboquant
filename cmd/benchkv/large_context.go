package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type ladderRejection struct {
	NumCtx int
	Stage  string
	Class  string
	Reason string
	Detail string
}

func isLargeContextWorkload(workload workloadName) bool {
	switch workload {
	case workloadFitCeiling, workloadLongContextRecall, workloadLongJSONRetention, workloadPromptFileRegress, workloadDecodeCorruption, workloadPrefillHeavy:
		return true
	default:
		return false
	}
}

func runLargeContextCell(cfg config, cell sweepCell, promptGen *promptGenerator, preflight hostPreflight, tracker *progressTracker) ([]workerResult, []epochAggregate, error) {
	var allWorkers []workerResult
	var allAggs []epochAggregate
	var rejected []ladderRejection
	requestedTop := cfg.StretchContext
	if requestedTop <= 0 && len(cfg.ContextLadder) > 0 {
		requestedTop = cfg.ContextLadder[0]
	}

	for idx, attempted := range cfg.ContextLadder {
		rungCell := cell
		rungCell.Workload.NumCtx = attempted
		spec, ok := makeWorkloadSpec(cfg, rungCell.Workload.Name, attempted, rungCell.Workload.Concurrency)
		if ok {
			rungCell.Workload = spec
		}
		cal, err := promptGen.promptForTarget(context.Background(), rungCell.Host, cfg.Model, rungCell.Workload.PromptTokensTarget, max(rungCell.Workload.NumCtx, 4096), cfg.Timeout)
		if err != nil {
			rej := ladderRejection{
				NumCtx: attempted,
				Stage:  "prefill",
				Class:  classifyContextFailure("prefill", err.Error()),
				Reason: err.Error(),
				Detail: err.Error(),
			}
			rejected = append(rejected, rej)
			continue
		}

		var rungWorkers []workerResult
		var rungAggs []epochAggregate
		success := true

		for warmup := 0; warmup < cfg.Warmup; warmup++ {
			if tracker != nil {
				tracker.SetCurrent(progressStep{
					Phase:       "ladder",
					HostLabel:   rungCell.Host.Label,
					KVMode:      rungCell.KVMode,
					Workload:    string(rungCell.Workload.Name),
					NumCtx:      attempted,
					Concurrency: rungCell.Workload.Concurrency,
					Epoch:       warmup + 1,
					EpochTotal:  cfg.Warmup,
					Warmup:      true,
				})
			}
			rows, agg := runEpoch(cfg, rungCell, preflight, cal, warmup+1, true)
			rows, agg = applyLargeContextMetadata(cfg, rungCell, rows, agg, requestedTop, attempted, idx, rejected)
			rungWorkers = append(rungWorkers, rows...)
			rungAggs = append(rungAggs, agg)
			if tracker != nil {
				tracker.Advance(1)
			}
			if agg.Status != statusOK {
				success = false
				rejected = append(rejected, classifyAggregateRejection(attempted, agg))
				break
			}
		}
		if !success {
			allWorkers = append(allWorkers, rungWorkers...)
			allAggs = append(allAggs, rungAggs...)
			continue
		}

		for epoch := 0; epoch < cfg.Epochs; epoch++ {
			if tracker != nil {
				tracker.SetCurrent(progressStep{
					Phase:       "ladder",
					HostLabel:   rungCell.Host.Label,
					KVMode:      rungCell.KVMode,
					Workload:    string(rungCell.Workload.Name),
					NumCtx:      attempted,
					Concurrency: rungCell.Workload.Concurrency,
					Epoch:       epoch + 1,
					EpochTotal:  cfg.Epochs,
				})
			}
			rows, agg := runEpoch(cfg, rungCell, preflight, cal, epoch+1, false)
			rows, agg = applyLargeContextMetadata(cfg, rungCell, rows, agg, requestedTop, attempted, idx, rejected)
			rungWorkers = append(rungWorkers, rows...)
			rungAggs = append(rungAggs, agg)
			if tracker != nil {
				tracker.Advance(1)
			}
			if agg.Status != statusOK || agg.EvalCount < cfg.MinFitDecode {
				success = false
				rej := classifyAggregateRejection(attempted, agg)
				if agg.Status == statusOK && agg.EvalCount < cfg.MinFitDecode {
					rej.Stage = "decode"
					rej.Class = "validation_failure"
					rej.Reason = fmt.Sprintf("minimum decode sample not reached: got %d want %d", agg.EvalCount, cfg.MinFitDecode)
					rej.Detail = rej.Reason
				}
				rejected = append(rejected, rej)
				break
			}
		}

		allWorkers = append(allWorkers, rungWorkers...)
		allAggs = append(allAggs, rungAggs...)
		if success {
			return allWorkers, allAggs, nil
		}
	}

	if len(rejected) == 0 {
		return allWorkers, allAggs, fmt.Errorf("large-context ladder failed without classified rejection")
	}
	last := rejected[len(rejected)-1]
	return allWorkers, allAggs, fmt.Errorf("large-context ladder exhausted at ctx=%d: %s", last.NumCtx, firstNonEmpty(last.Reason, last.Detail))
}

func applyLargeContextMetadata(cfg config, cell sweepCell, rows []workerResult, agg epochAggregate, requestedTop, attempted, ladderIndex int, rejected []ladderRejection) ([]workerResult, epochAggregate) {
	rejectedSummary := formatRejectedRungs(rejected)
	for i := range rows {
		rows[i] = enrichLargeContextRow(cfg, cell, rows[i], requestedTop, attempted, ladderIndex, rejectedSummary)
	}
	agg = enrichLargeContextAggregate(cfg, cell, agg, requestedTop, attempted, ladderIndex, rejectedSummary)
	return rows, agg
}

func enrichLargeContextRow(cfg config, cell sweepCell, row workerResult, requestedTop, attempted, ladderIndex int, rejectedSummary string) workerResult {
	row.RequestedNumCtx = requestedTop
	row.AttemptedNumCtx = attempted
	row.EffectiveNumCtx = attempted
	if row.ContextLength > 0 {
		row.EffectiveNumCtx = row.ContextLength
	}
	row.RequestedContextTopRung = requestedTop
	row.ContextLadderIndex = ladderIndex
	row.LadderRejectedRungs = rejectedSummary
	row.ModelFileSizeBytes = int64Ptr(row.SizeBytes)
	row.LongContextCapSource = detectLongContextCapSource(cfg, row)
	row.NativeContextAdvertised = detectNativeContextAdvertised(cfg, row)
	row.YarnContextAdvertised = detectYarnContextAdvertised(cfg)
	row.EstimatedKVFootprintBytes = estimateKVFootprint(row)
	row.UsedHostAssist, row.UsedMMap, row.UsedCPUAssist = deriveAssistFlags(row)
	if row.Status != statusOK {
		row.ContextFallbackStage = classifyFallbackStage(row.Error)
		row.ContextFallbackClass = classifyContextFailure(row.ContextFallbackStage, row.Error)
		row.ContextFallbackReason = row.Error
		row.ContextFallbackDetail = row.Error
	} else if row.FallbackReason != "" {
		row.ContextFallbackReason = row.FallbackReason
	}
	return row
}

func enrichLargeContextAggregate(cfg config, cell sweepCell, agg epochAggregate, requestedTop, attempted, ladderIndex int, rejectedSummary string) epochAggregate {
	agg.RequestedNumCtx = requestedTop
	agg.AttemptedNumCtx = attempted
	agg.EffectiveNumCtx = attempted
	if agg.NumCtx > 0 {
		agg.EffectiveNumCtx = agg.NumCtx
	}
	agg.RequestedContextTopRung = requestedTop
	agg.ContextLadderIndex = ladderIndex
	agg.LadderRejectedRungs = rejectedSummary
	if agg.RunnerRSSBytes > 0 {
		agg.ModelFileSizeBytes = int64Ptr(agg.RunnerRSSBytes)
	}
	agg.LongContextCapSource = detectLongContextCapSource(cfg, workerResult{ContextLength: agg.EffectiveNumCtx})
	agg.NativeContextAdvertised = detectNativeContextAdvertised(cfg, workerResult{ContextLength: agg.EffectiveNumCtx})
	agg.YarnContextAdvertised = detectYarnContextAdvertised(cfg)
	if agg.Status != statusOK {
		agg.ContextFallbackStage = classifyFallbackStage(agg.Error)
		agg.ContextFallbackClass = classifyContextFailure(agg.ContextFallbackStage, agg.Error)
		agg.ContextFallbackReason = agg.Error
		agg.ContextFallbackDetail = agg.Error
	} else if agg.FallbackReason != "" {
		agg.ContextFallbackReason = agg.FallbackReason
	}
	return agg
}

func classifyAggregateRejection(numCtx int, agg epochAggregate) ladderRejection {
	stage := classifyFallbackStage(agg.Error)
	if stage == "" {
		stage = "decode"
	}
	reason := firstNonEmpty(agg.Error, agg.FallbackReason)
	return ladderRejection{
		NumCtx: numCtx,
		Stage:  stage,
		Class:  classifyContextFailure(stage, reason),
		Reason: reason,
		Detail: reason,
	}
}

func classifyFallbackStage(message string) string {
	lower := strings.ToLower(strings.TrimSpace(message))
	switch {
	case lower == "":
		return ""
	case strings.Contains(lower, "timeout"):
		return "timeout"
	case strings.Contains(lower, "prompt_eval") || strings.Contains(lower, "prefill"):
		return "prefill"
	case strings.Contains(lower, "context") && strings.Contains(lower, "refus"):
		return "backend_refusal"
	case strings.Contains(lower, "load"):
		return "load"
	default:
		return "decode"
	}
}

func classifyContextFailure(stage, message string) string {
	lower := strings.ToLower(strings.TrimSpace(message))
	switch {
	case strings.Contains(lower, "timeout") || stage == "timeout":
		return "timeout"
	case strings.Contains(lower, "context") && (strings.Contains(lower, "maximum") || strings.Contains(lower, "too large") || strings.Contains(lower, "unsupported")):
		return "runtime_context_cap"
	case strings.Contains(lower, "out of memory") || strings.Contains(lower, "cuda error") || strings.Contains(lower, "vram"):
		return "vram_exhausted"
	case strings.Contains(lower, "cannot allocate memory") || strings.Contains(lower, "host ram") || strings.Contains(lower, "system memory"):
		return "host_ram_exhausted"
	case strings.Contains(lower, "spill") || (strings.Contains(lower, "memory") && strings.Contains(lower, "gpu")):
		return "mixed_memory_pressure"
	case strings.Contains(lower, "unsupported") || stage == "backend_refusal":
		return "backend_refusal"
	case strings.Contains(lower, "validation"):
		return "validation_failure"
	default:
		return "unknown"
	}
}

func formatRejectedRungs(rejected []ladderRejection) string {
	if len(rejected) == 0 {
		return ""
	}
	parts := make([]string, 0, len(rejected))
	for _, rej := range rejected {
		parts = append(parts, fmt.Sprintf("%d:%s:%s", rej.NumCtx, rej.Stage, firstNonEmpty(rej.Class, "unknown")))
	}
	return strings.Join(parts, ";")
}

func detectLongContextCapSource(cfg config, row workerResult) string {
	switch {
	case cfg.TargetYarnCtx > 0:
		return "flag:yarn"
	case cfg.TargetNativeCtx > 0:
		return "flag:native"
	case row.ContextLength > 0:
		return "runtime:list-running"
	default:
		return "unknown"
	}
}

func detectNativeContextAdvertised(cfg config, row workerResult) int {
	if cfg.TargetNativeCtx > 0 {
		return cfg.TargetNativeCtx
	}
	return row.ContextLength
}

func detectYarnContextAdvertised(cfg config) int {
	if cfg.TargetYarnCtx > 0 {
		return cfg.TargetYarnCtx
	}
	return 0
}

func estimateKVFootprint(row workerResult) *int64 {
	if row.EffectiveNumCtx <= 0 || row.SizeBytes <= 0 {
		return nil
	}
	estimate := int64(row.EffectiveNumCtx) * int64(max(row.DetectedHeadDim, 1)) * 8
	if estimate <= 0 {
		return nil
	}
	return int64Ptr(estimate)
}

func deriveAssistFlags(row workerResult) (bool, bool, bool) {
	usedCPU := strings.Contains(strings.ToLower(row.ProcessorStateAfter), "cpu") || strings.Contains(strings.ToLower(row.ProcessorStateAfter), "partial gpu")
	usedHost := row.Spilled || row.GPUOffloadRegression || ptrInt64Value(row.PeakHostRAMDeltaBytes) > 0 || (row.SizeBytes > 0 && row.SizeVRAMBytes > 0 && row.SizeBytes > row.SizeVRAMBytes)
	usedMMap := usedHost && row.SizeBytes > row.SizeVRAMBytes
	return usedHost, usedMMap, usedCPU
}

func buildWorkerPrompt(cfg config, cell sweepCell, cal promptCalibration, epoch, workerIndex int) (string, error) {
	switch cell.Workload.Name {
	case workloadLongContextRecall:
		return buildLongContextRecallPrompt(cell.Workload.PromptTokensTarget), nil
	case workloadPromptFileRegress:
		if strings.TrimSpace(cfg.PromptFile) != "" {
			data, err := os.ReadFile(cfg.PromptFile)
			if err != nil {
				return "", err
			}
			return string(data), nil
		}
		return buildPromptFileRegressionPrompt(cell.Workload.PromptTokensTarget), nil
	case workloadDecodeCorruption:
		return buildDecodeCorruptionPrompt(cell.Workload.PromptTokensTarget), nil
	case workloadLongJSONRetention:
		return buildLongJSONRetentionPrompt(max(cfg.LongJSONBytes, 1024)), nil
	default:
		return renderWorkerPrompt(cal, epoch, workerIndex), nil
	}
}

func buildLongContextRecallPrompt(targetTokens int) string {
	const exactNeedle = "LC-NEEDLE-7781"
	const numericNeedle = "48291057"
	filler := strings.Repeat("filler span ", max(targetTokens/3, 512))
	return "Read the long context carefully and answer with the exact string token and numeric token only.\n" +
		filler + "\nSTRING NEEDLE: " + exactNeedle + "\n" +
		filler + "\nNUMERIC NEEDLE: " + numericNeedle + "\n" +
		filler + "\nReply in the form string=<token> number=<digits>."
}

func buildPromptFileRegressionPrompt(targetTokens int) string {
	repeated := strings.Repeat("prompt-file regression coverage sentence with punctuation / ? and JSON-like braces {} [] ", max(targetTokens/8, 64))
	return repeated + "\nSummarize the repeated phrase in one sentence and keep punctuation stable."
}

func buildDecodeCorruptionPrompt(targetTokens int) string {
	filler := strings.Repeat("stability context ", max(targetTokens/4, 256))
	return filler + "\nReturn valid JSON with keys answer and checksum where checksum is 12345."
}

func buildLongJSONRetentionPrompt(targetBytes int) string {
	var b strings.Builder
	b.WriteString("{\"records\":[")
	records := max(targetBytes/64, 64)
	for i := 0; i < records; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString("{\"id\":")
		b.WriteString(strconv.Itoa(1000 + i))
		b.WriteString(",\"name\":\"entry-")
		b.WriteString(strconv.Itoa(i))
		b.WriteString("\",\"value\":\"payload\"}")
	}
	b.WriteString("]}\nAnswer with the id for entry-")
	b.WriteString(strconv.Itoa(records / 2))
	b.WriteString(" only.")
	return b.String()
}
