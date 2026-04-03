package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/tqbenchschema"
	"github.com/ollama/ollama/turboquant"
)

func runBenchmark(cfg config, tracker *progressTracker) ([]workerResult, []epochAggregate, []staircaseRecord, error) {
	promptGen := newPromptGenerator()
	preflights := make(map[string]hostPreflight)
	for _, host := range cfg.Hosts {
		if tracker != nil {
			tracker.Status(fmt.Sprintf("preflight: %s", host.Label))
			tracker.SetCurrent(progressStep{Phase: "preflight", HostLabel: host.Label, Epoch: 1, EpochTotal: 1})
		}
		ctx, cancel := withOptionalTimeout(context.Background(), preflightTimeout(cfg.Timeout))
		preflight, err := preflightHost(ctx, cfg, host)
		cancel()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("preflight failed for %s: %w", host.Label, err)
		}
		preflights[host.BaseURL] = preflight
		if tracker != nil {
			tracker.Status(fmt.Sprintf("preflight complete: %s", host.Label))
			tracker.Advance(1)
		}
	}
	if tracker != nil {
		tracker.SetTotal(estimateTotalUnitsWithPreflight(cfg, preflights))
	}

	var workers []workerResult
	var aggregates []epochAggregate
	var staircases []staircaseRecord

	for _, cell := range buildStandardCells(cfg) {
		preflight := preflights[cell.Host.BaseURL]
		supported := preflight.KVSupport[cell.KVMode]
		if !supported.Supported {
			warmupRow, agg := unsupportedRow(cfg, cell, preflight, supported.Error)
			workers = append(workers, warmupRow)
			aggregates = append(aggregates, agg)
			if cfg.FailFast {
				return workers, aggregates, staircases, errors.New(supported.Error)
			}
			continue
		}

		cellWorkers, cellAggs, err := runCell(cfg, cell, promptGen, preflight, tracker)
		workers = append(workers, cellWorkers...)
		aggregates = append(aggregates, cellAggs...)
		if err != nil && cfg.FailFast {
			return workers, aggregates, staircases, err
		}
	}

	if slices.Contains(cfg.Workloads, workloadNearOOMStaircase) {
		for _, host := range cfg.Hosts {
			if (cfg.Profile == "capacity" || cfg.Profile == "spill") && host.KVSupportMode != hostKVSupportRequest {
				continue
			}
			preflight := preflights[host.BaseURL]
			for _, kvMode := range cfg.KVModes {
				record, cellWorkers, cellAggs, err := runStaircase(cfg, host, kvMode, promptGen, preflight, tracker)
				workers = append(workers, cellWorkers...)
				aggregates = append(aggregates, cellAggs...)
				staircases = append(staircases, record)
				if err != nil && cfg.FailFast {
					return workers, aggregates, staircases, err
				}
			}
		}
	}

	return workers, aggregates, staircases, nil
}

func preflightTimeout(runTimeout time.Duration) time.Duration {
	if runTimeout <= 0 {
		return 0
	}
	if runTimeout < 2*time.Minute {
		return 2 * time.Minute
	}
	return runTimeout
}

func unsupportedRow(cfg config, cell sweepCell, preflight hostPreflight, errText string) (workerResult, epochAggregate) {
	requestedK, requestedV, _ := splitBenchmarkKVMode(cell.KVMode)
	row := workerResult{
		Host:                    cell.Host.BaseURL,
		HostLabel:               cell.Host.Label,
		ServerVersion:           preflight.Version,
		Model:                   cfg.Model,
		ModelFamily:             firstNonEmpty(cfg.ModelFamilyOverride, preflight.ModelFamily),
		ModelArch:               "unknown",
		ModelSizeLabel:          deriveModelSizeLabel(cfg.Model, cfg.ModelSizeLabel),
		Quant:                   preflight.ModelQuant,
		KVModeRequested:         cell.KVMode,
		KVModeRequestedK:        requestedK,
		KVModeRequestedV:        requestedV,
		RequestedCacheTypeK:     requestedK,
		RequestedCacheTypeV:     requestedV,
		SymmetricRequested:      strings.EqualFold(requestedK, requestedV),
		FlashAttentionRequested: cell.FARequested,
		RequestedMode:           summarizeRequestedOrEffectiveMode(requestedK, requestedV),
		EffectiveMode:           summarizeRequestedOrEffectiveMode(requestedK, requestedV),
		KVBackendRequested:      requestedKVBackend(cell.KVMode),
		KVAlgoResolved:          firstNonEmpty(kvAlgoForRequestedMode(cell.KVMode), "unknown"),
		KVPath:                  "unknown",
		Workload:                string(cell.Workload.Name),
		NumCtx:                  cell.Workload.NumCtx,
		PromptTokensTarget:      cell.Workload.PromptTokensTarget,
		MaxTokens:               cell.Workload.MaxTokens,
		CtxXConc:                cell.Workload.NumCtx * cell.Workload.Concurrency,
		Concurrency:             cell.Workload.Concurrency,
		Epoch:                   0,
		Warmup:                  false,
		RunnerRSSBytes:          -1,
		GPUResidency:            "unknown",
		ResidencyKind:           string(tqbenchschema.ResidencyUnknown),
		FitStatus:               string(tqbenchschema.FitStatusUnsupported),
		CorrectnessStatus:       string(tqbenchschema.CorrectnessStatusUnsupported),
		CorruptionStatus:        string(tqbenchschema.CorruptionStatusSkipped),
		Status:                  statusUnsupported,
		Success:                 nil,
		Error:                   errText,
		RecordedAt:              time.Now().UTC(),
	}
	row.RequestedNumCtx = cell.Workload.NumCtx
	row.AttemptedNumCtx = cell.Workload.NumCtx
	row.EffectiveNumCtx = cell.Workload.NumCtx
	row.ContextRequested = cell.Workload.NumCtx
	row.ContextEffective = cell.Workload.NumCtx
	applyDerivedStatuses(&row)
	agg := epochAggregate{
		Host:               row.Host,
		HostLabel:          row.HostLabel,
		ServerVersion:      row.ServerVersion,
		Model:              row.Model,
		Quant:              row.Quant,
		KVModeRequested:    row.KVModeRequested,
		KVModeRequestedK:   row.KVModeRequestedK,
		KVModeRequestedV:   row.KVModeRequestedV,
		RequestedMode:      row.RequestedMode,
		EffectiveMode:      row.EffectiveMode,
		KVAlgoResolved:     row.KVAlgoResolved,
		KVBackendRequested: row.KVBackendRequested,
		KVPath:             row.KVPath,
		Workload:           row.Workload,
		NumCtx:             row.NumCtx,
		PromptTokensTarget: row.PromptTokensTarget,
		MaxTokens:          row.MaxTokens,
		CtxXConc:           row.CtxXConc,
		Concurrency:        row.Concurrency,
		Epoch:              0,
		Warmup:             false,
		RunnerRSSBytes:     row.RunnerRSSBytes,
		Status:             statusUnsupported,
		Success:            nil,
		Error:              errText,
	}
	return row, agg
}

func runCell(cfg config, cell sweepCell, promptGen *promptGenerator, preflight hostPreflight, tracker *progressTracker) ([]workerResult, []epochAggregate, error) {
	if cfg.Profile == "large-context" || isLargeContextWorkload(cell.Workload.Name) {
		return runLargeContextCell(cfg, cell, promptGen, preflight, tracker)
	}
	cal, err := promptGen.promptForTarget(context.Background(), cell.Host, cfg.Model, cell.Workload.PromptTokensTarget, max(cell.Workload.NumCtx, 4096), cfg.Timeout)
	if err != nil {
		return nil, nil, err
	}

	var allWorkers []workerResult
	var allAggs []epochAggregate

	for warmup := 0; warmup < cfg.Warmup; warmup++ {
		if tracker != nil {
			tracker.SetCurrent(progressStep{
				Phase:       "epoch",
				HostLabel:   cell.Host.Label,
				KVMode:      cell.KVMode,
				Workload:    string(cell.Workload.Name),
				NumCtx:      cell.Workload.NumCtx,
				Concurrency: cell.Workload.Concurrency,
				Epoch:       warmup + 1,
				EpochTotal:  cfg.Warmup,
				Warmup:      true,
			})
		}
		rows, agg := runEpoch(cfg, cell, preflight, cal, warmup+1, true)
		if agg.Status != statusOK {
			if tracker != nil {
				tracker.Status(fmt.Sprintf("retrying warmup: %s %s %s ctx=%d conc=%d warmup=%d", cell.Host.Label, cell.KVMode, cell.Workload.Name, cell.Workload.NumCtx, cell.Workload.Concurrency, warmup+1))
			}
			retryRows, retryAgg := runEpoch(cfg, cell, preflight, cal, warmup+1, true)
			allWorkers = append(allWorkers, rows...)
			allAggs = append(allAggs, agg)
			allWorkers = append(allWorkers, retryRows...)
			allAggs = append(allAggs, retryAgg)
			if tracker != nil {
				tracker.Advance(1)
			}
			if retryAgg.Status != statusOK {
				return allWorkers, allAggs, fmt.Errorf("warmup failed for %s %s ctx=%d conc=%d: %s", cell.Host.Label, cell.KVMode, cell.Workload.NumCtx, cell.Workload.Concurrency, retryAgg.Error)
			}
			continue
		}
		allWorkers = append(allWorkers, rows...)
		allAggs = append(allAggs, agg)
		if tracker != nil {
			tracker.Advance(1)
		}
	}
	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		if tracker != nil {
			tracker.SetCurrent(progressStep{
				Phase:       "epoch",
				HostLabel:   cell.Host.Label,
				KVMode:      cell.KVMode,
				Workload:    string(cell.Workload.Name),
				NumCtx:      cell.Workload.NumCtx,
				Concurrency: cell.Workload.Concurrency,
				Epoch:       epoch + 1,
				EpochTotal:  cfg.Epochs,
			})
		}
		rows, agg := runEpoch(cfg, cell, preflight, cal, epoch+1, false)
		allWorkers = append(allWorkers, rows...)
		allAggs = append(allAggs, agg)
		if tracker != nil {
			if agg.Status != statusOK {
				tracker.Status(fmt.Sprintf("failed: %s %s %s ctx=%d conc=%d epoch=%d (%s)", cell.Host.Label, cell.KVMode, cell.Workload.Name, cell.Workload.NumCtx, cell.Workload.Concurrency, epoch+1, agg.Error))
			}
			tracker.Advance(1)
		}
	}

	return allWorkers, allAggs, nil
}

func runStaircase(cfg config, host hostTarget, kvMode string, promptGen *promptGenerator, preflight hostPreflight, tracker *progressTracker) (staircaseRecord, []workerResult, []epochAggregate, error) {
	record := staircaseRecord{
		HostLabel: host.Label,
		Host:      host.BaseURL,
		KVMode:    kvMode,
	}
	if result := preflight.KVSupport[kvMode]; !result.Supported {
		record.FailureReason = result.Error
		if len(cfg.NumCtx) > 0 && len(cfg.Concurrency) > 0 {
			if spec, ok := makeWorkloadSpec(cfg, workloadNearOOMStaircase, cfg.NumCtx[0], cfg.Concurrency[0]); ok {
				row, agg := unsupportedRow(cfg, sweepCell{Host: host, KVMode: kvMode, Workload: spec}, preflight, result.Error)
				return record, []workerResult{row}, []epochAggregate{agg}, nil
			}
		}
		return record, nil, nil, nil
	}

	var workers []workerResult
	var aggs []epochAggregate

	stopOnSpill := cfg.Profile == "capacity"
	for _, numCtx := range cfg.NumCtx {
		for _, conc := range cfg.Concurrency {
			spec, ok := makeWorkloadSpec(cfg, workloadNearOOMStaircase, numCtx, conc)
			if !ok {
				continue
			}
			cell := sweepCell{Host: host, KVMode: kvMode, Workload: spec}
			cellWorkers, cellAggs, err := runCell(cfg, cell, promptGen, preflight, tracker)
			workers = append(workers, cellWorkers...)
			aggs = append(aggs, cellAggs...)

			stop := false
			for _, agg := range cellAggs {
				if agg.Warmup {
					continue
				}
				if agg.LiveKVTokensTotal > record.MaxLiveKVTokensTotal {
					record.MaxLiveKVTokensTotal = agg.LiveKVTokensTotal
				}
				if agg.Status != statusOK {
					stop = true
					record.FailedNumCtx = numCtx
					record.FailedConc = conc
					record.FirstFailureCtxXConcurrency = numCtx * conc
					record.FailureReason = agg.Error
					break
				}
				if agg.Spilled && record.FirstSpillCtx == 0 {
					record.FirstSpillCtx = numCtx
					record.FirstSpillCtxXConcurrency = numCtx * conc
				}
				if stopOnSpill && (agg.Spilled || agg.GPUOffloadRegression || !agg.FullGPUResidency) {
					stop = true
					record.FailedNumCtx = numCtx
					record.FailedConc = conc
					if record.FirstSpillCtx == 0 {
						record.FirstSpillCtx = numCtx
						record.FirstSpillCtxXConcurrency = numCtx * conc
					}
					record.FailureReason = "spill/offload detected"
					break
				}
			}
			if stop {
				if tracker != nil {
					remaining := remainingStaircaseUnits(cfg, conc, numCtx)
					if remaining > 0 {
						tracker.Skip(remaining, fmt.Sprintf("staircase stop: %s %s last_stable=(ctx=%d,conc=%d) failed=(ctx=%d,conc=%d)", host.Label, kvMode, record.LastStableNumCtx, record.LastStableConc, record.FailedNumCtx, record.FailedConc))
					}
				}
				return record, workers, aggs, err
			}
			record.LastStableNumCtx = numCtx
			record.LastStableConc = conc
			record.MaxStableCtxXConcurrency = numCtx * conc
			if timedAggregateWithFullGPU(cellAggs) {
				if conc == 1 && numCtx > record.MaxFullGPUCtx {
					record.MaxFullGPUCtx = numCtx
				}
				record.LastFullGPUCtxXConcurrency = numCtx * conc
			}
		}
	}

	return record, workers, aggs, nil
}

func remainingStaircaseUnits(cfg config, failedConc, failedNumCtx int) int {
	unitsPerCell := cfg.Warmup + cfg.Epochs
	if unitsPerCell <= 0 {
		return 0
	}

	seenFailure := false
	remainingCells := 0
	for _, numCtx := range cfg.NumCtx {
		for _, conc := range cfg.Concurrency {
			if _, ok := makeWorkloadSpec(cfg, workloadNearOOMStaircase, numCtx, conc); !ok {
				continue
			}
			if seenFailure {
				remainingCells++
				continue
			}
			if conc == failedConc && numCtx == failedNumCtx {
				seenFailure = true
			}
		}
	}

	return remainingCells * unitsPerCell
}

func runEpoch(cfg config, cell sweepCell, preflight hostPreflight, cal promptCalibration, epoch int, warmup bool) ([]workerResult, epochAggregate) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	beforePS := ""
	if cfg.CaptureOllamaPS {
		beforePS = captureOllamaPS()
	}

	monitor := newGPUMonitor(cfg.ProbeInterval, cfg.CaptureGPU)
	hostMonitor := newHostMetricsMonitor(cfg.ProbeInterval, true)
	go monitor.run(ctx)
	go hostMonitor.run(ctx)

	start := time.Now()
	results := make([]workerResult, cell.Workload.Concurrency)
	var wg sync.WaitGroup
	for workerIndex := 0; workerIndex < cell.Workload.Concurrency; workerIndex++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			results[idx] = runWorker(cfg, cell, preflight, cal, epoch, warmup, idx)
		}(workerIndex)
	}
	wg.Wait()
	cancel()

	stats := monitor.stats()
	hostStats := hostMonitor.stats()
	afterPS := ""
	if cfg.CaptureOllamaPS {
		afterPS = captureOllamaPS()
	}
	processorBefore := processorState(beforePS)
	processorAfter := processorState(afterPS)
	spilled := spilledState(processorAfter)
	offloadRegression := gpuOffloadRegression(beforePS, afterPS)
	for i := range results {
		results[i].ServerVersion = preflight.Version
		results[i].Quant = preflight.ModelQuant
		results[i].GPUMetricsAvailable = stats.Available
		results[i].GPUStatsSource = stats.Source
		results[i].PeakVRAMBytes = stats.PeakVRAMBytes
		results[i].AvgGPUUtil = stats.AvgGPUUtil
		results[i].PeakGPUUtil = stats.PeakGPUUtil
		results[i].VisibleGPUCount = stats.VisibleGPUCount
		results[i].TotalVisibleVRAMBytes = stats.TotalVisibleVRAMBytes
		results[i].ProcessVRAMBytes = stats.ProcessVRAMBytes
		results[i].GPUVRAMUsedBytes = stats.PeakVRAMBytes
		results[i].GPUVRAMFreeBytes = computeGPUVRAMFree(stats.TotalVisibleVRAMBytes, stats.PeakVRAMBytes)
		results[i].PerGPUVRAMGiB = formatPerGPUVRAMGiB(stats.PerGPUUsedBytes)
		results[i].HostMetricsAvailable = hostStats.Available
		results[i].HostStatsSource = hostStats.Source
		results[i].HostRAMBeforeBytes = hostStats.HostRAMBeforeBytes
		results[i].HostRAMUsedBytes = hostStats.HostRAMUsedBytes
		results[i].PeakHostRAMBytes = hostStats.PeakHostRAMBytes
		results[i].PeakHostRAMDeltaBytes = hostStats.PeakHostRAMDeltaBytes
		results[i].HostRAMAfterDecodeBytes = hostStats.HostRAMUsedBytes
		results[i].HostRAMAfterPrefillBytes = hostStats.PeakHostRAMBytes
		results[i].HostRAMAfterLoadBytes = hostStats.HostRAMBeforeBytes
		if hostStats.ProcessRSSBytes != nil {
			results[i].RunnerRSSBytes = *hostStats.ProcessRSSBytes
		} else if cfg.CaptureRunnerRSS {
			results[i].RunnerRSSBytes = captureRunnerRSS()
		} else {
			results[i].RunnerRSSBytes = -1
		}
		if cfg.CaptureOllamaPS {
			results[i].ProcessorStateBefore = processorBefore
			results[i].ProcessorStateAfter = processorAfter
			results[i].GPUResidency = processorAfter
			results[i].FullGPUResidency = fullGPUResidency(afterPS)
			results[i].Spilled = spilled
			results[i].GPUOffloadRegression = offloadRegression
		}
		results[i].UsedHostAssist, results[i].UsedMMap, results[i].UsedCPUAssist = deriveAssistFlags(results[i])
		results[i].ResidencyKind = string(deriveResidencyKind(results[i]))
		applyDerivedStatuses(&results[i])
	}

	return results, aggregateEpoch(results, time.Since(start))
}

func runWorker(cfg config, cell sweepCell, preflight hostPreflight, cal promptCalibration, epoch int, warmup bool, workerIndex int) workerResult {
	requestedK, requestedV, _ := splitBenchmarkKVMode(cell.KVMode)
	row := workerResult{
		Host:                    cell.Host.BaseURL,
		HostLabel:               cell.Host.Label,
		Model:                   cfg.Model,
		ModelFamily:             firstNonEmpty(cfg.ModelFamilyOverride, preflight.ModelFamily),
		ModelArch:               "unknown",
		ModelSizeLabel:          deriveModelSizeLabel(cfg.Model, cfg.ModelSizeLabel),
		KVModeRequested:         cell.KVMode,
		KVModeRequestedK:        requestedK,
		KVModeRequestedV:        requestedV,
		RequestedCacheTypeK:     requestedK,
		RequestedCacheTypeV:     requestedV,
		KVBackendRequested:      requestedKVBackend(cell.KVMode),
		RequestedMode:           summarizeRequestedOrEffectiveMode(requestedK, requestedV),
		SymmetricRequested:      strings.EqualFold(requestedK, requestedV),
		FlashAttentionRequested: cell.FARequested,
		Workload:                string(cell.Workload.Name),
		NumCtx:                  cell.Workload.NumCtx,
		RequestedNumCtx:         cell.Workload.NumCtx,
		AttemptedNumCtx:         cell.Workload.NumCtx,
		ContextRequested:        cell.Workload.NumCtx,
		RequestedContextTopRung: cell.Workload.NumCtx,
		PromptTokensTarget:      cell.Workload.PromptTokensTarget,
		MaxTokens:               cell.Workload.MaxTokens,
		CtxXConc:                cell.Workload.NumCtx * cell.Workload.Concurrency,
		Concurrency:             cell.Workload.Concurrency,
		WorkerIndex:             workerIndex,
		Epoch:                   epoch,
		Warmup:                  warmup,
		RunnerRSSBytes:          -1,
		GPUResidency:            "unknown",
		GPUStatsSource:          "unavailable",
		HostStatsSource:         "unavailable",
		ResidencyKind:           string(tqbenchschema.ResidencyUnknown),
		FitStatus:               string(tqbenchschema.FitStatusFailed),
		CorruptionStatus:        string(tqbenchschema.CorruptionStatusSkipped),
		CorrectnessStatus:       string(tqbenchschema.CorrectnessStatusSkipped),
		Status:                  statusFailed,
		RecordedAt:              time.Now().UTC(),
	}

	prompt, promptErr := buildWorkerPrompt(cfg, cell, cal, epoch, workerIndex)
	if promptErr != nil {
		row.Status = statusFailed
		row.Success = boolPtr(false)
		row.Error = promptErr.Error()
		return row
	}
	stream := cfg.Stream
	keepAlive := api.Duration{Duration: cfg.KeepAlive}
	options, disposition := buildGenerateOptions(cell.Host, cell.KVMode, cell.Workload.NumCtx, cell.Workload.MaxTokens, cfg.Seed, cfg.Temperature)
	if !disposition.Supported {
		row.Status = statusUnsupported
		row.Success = nil
		row.Error = disposition.Error
		return row
	}
	applyFlashAttentionOption(options, cell.FARequested)

	req := &api.GenerateRequest{
		Model:     cfg.Model,
		Prompt:    prompt,
		Raw:       true,
		Stream:    &stream,
		KeepAlive: &keepAlive,
		Options:   options,
	}

	ctx, cancel := withOptionalTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	requestStart := time.Now()
	var ttft time.Duration
	var ttftOnce sync.Once
	var finalMetrics *api.Metrics

	err := cell.Host.Client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		ttftOnce.Do(func() {
			if strings.TrimSpace(resp.Response) != "" || strings.TrimSpace(resp.Thinking) != "" {
				ttft = time.Since(requestStart)
			}
		})
		if resp.Done {
			copy := resp.Metrics
			finalMetrics = &copy
		}
		return nil
	})
	row.WallMS = float64(time.Since(requestStart)) / float64(time.Millisecond)
	row.WallTimeS = row.WallMS / 1000
	if err != nil {
		row.Status = statusFailed
		row.Success = boolPtr(false)
		row.Error = err.Error()
		if isUnsupportedKVError(err) {
			row.Status = statusUnsupported
			row.Success = nil
			row.Error = "unsupported kv_cache_type on target host"
		}
		return enrichWithRunningInfo(ctx, cell.Host.Client, cfg.Model, row)
	}
	if finalMetrics == nil {
		row.Status = statusFailed
		row.Success = boolPtr(false)
		row.Error = "no final metrics received"
		return enrichWithRunningInfo(ctx, cell.Host.Client, cfg.Model, row)
	}

	row.Status = statusOK
	row.Success = boolPtr(true)
	row.EffectiveNumCtx = cell.Workload.NumCtx
	row.KVModeResolved = firstNonEmpty(finalMetrics.ResolvedKVCacheType, finalMetrics.KVCacheEffective, row.KVModeRequested)
	row.KVModeResolvedK = firstNonEmpty(finalMetrics.ResolvedKVCacheTypeK, row.KVModeRequestedK)
	row.KVModeResolvedV = firstNonEmpty(finalMetrics.ResolvedKVCacheTypeV, row.KVModeRequestedV)
	row.EffectiveCacheTypeK = firstNonEmpty(row.KVModeResolvedK, row.KVModeRequestedK)
	row.EffectiveCacheTypeV = firstNonEmpty(row.KVModeResolvedV, row.KVModeRequestedV)
	row.RequestedMode = firstNonEmpty(finalMetrics.RequestedMode, summarizeRequestedOrEffectiveMode(row.KVModeRequestedK, row.KVModeRequestedV))
	row.EffectiveMode = firstNonEmpty(finalMetrics.EffectiveMode, summarizeRequestedOrEffectiveMode(row.KVModeResolvedK, row.KVModeResolvedV), row.KVModeResolved)
	row.KVAlgoResolved = firstNonEmpty(finalMetrics.KVAlgoResolved, kvAlgoForRequestedMode(row.KVModeResolved))
	row.KVAlgoResolvedK = firstNonEmpty(finalMetrics.KVAlgoResolvedK, row.KVAlgoResolved)
	row.KVAlgoResolvedV = firstNonEmpty(finalMetrics.KVAlgoResolvedV, row.KVAlgoResolved)
	row.KVPath = firstNonEmpty(finalMetrics.KVCachePath, "unknown")
	row.KVPathK = firstNonEmpty(finalMetrics.KVCachePathK, row.KVPath)
	row.KVPathV = firstNonEmpty(finalMetrics.KVCachePathV, row.KVPath)
	row.KVSymmetric = finalMetrics.KVSymmetric
	row.KVAsymmetric = finalMetrics.KVAsymmetric
	row.SymmetricEffective = !row.KVAsymmetric
	row.FallbackApplied = finalMetrics.FallbackApplied
	row.KOnlyFallback = finalMetrics.KOnlyFallback
	row.FallbackReason = finalMetrics.FallbackReason
	row.TurboQuantPathKind = finalMetrics.TurboQuantPathKind
	row.PathKind = firstNonEmpty(finalMetrics.TurboQuantPathKind, finalMetrics.KVCachePath, "unknown")
	row.NativeTurboQuantActive = finalMetrics.NativeTurboQuantActive
	row.ReferenceTurboQuantActive = finalMetrics.ReferenceTurboQuantActive
	row.FAEnabled = finalMetrics.FAEnabled
	row.FlashAttentionEffective = finalMetrics.FAEnabled
	row.FARequiredForVTurbo = finalMetrics.FARequiredForVTurbo
	row.VTurboSupported = finalMetrics.VTurboSupported
	row.DetectedHeadDim = finalMetrics.DetectedHeadDim
	row.ArchitectureClass = finalMetrics.ArchitectureClass
	row.ModelArch = firstNonEmpty(finalMetrics.ArchitectureClass, row.ModelArch)
	row.SupportTier = finalMetrics.SupportTier
	row.HybridKVArchitecture = finalMetrics.HybridKVArchitecture
	row.TQBlockSize = finalMetrics.TQBlockSize
	row.PromptEvalCount = finalMetrics.PromptEvalCount
	row.PromptTokens = finalMetrics.PromptEvalCount
	row.EvalCount = finalMetrics.EvalCount
	row.GeneratedTokens = finalMetrics.EvalCount
	row.LiveKVTokensTotal = row.PromptEvalCount + row.GeneratedTokens
	row.PromptEvalMS = float64(finalMetrics.PromptEvalDuration) / float64(time.Millisecond)
	row.EvalMS = float64(finalMetrics.EvalDuration) / float64(time.Millisecond)
	row.PrefillTPS = tokensPerSecond(finalMetrics.PromptEvalCount, finalMetrics.PromptEvalDuration)
	row.DecodeTPS = tokensPerSecond(finalMetrics.EvalCount, finalMetrics.EvalDuration)
	row.TTFTMS = float64(ttft) / float64(time.Millisecond)
	row.LoadMS = float64(finalMetrics.LoadDuration) / float64(time.Millisecond)
	row.TotalMS = float64(finalMetrics.TotalDuration) / float64(time.Millisecond)
	if finalMetrics.KVCacheBytes > 0 {
		kvBytes := int64(finalMetrics.KVCacheBytes)
		row.KVBufferBytesEstimate = &kvBytes
		row.EstimatedKVFootprintBytes = &kvBytes
	}
	row.ContextEffective = max(finalMetrics.PromptEvalCount+finalMetrics.EvalCount, row.EffectiveNumCtx)
	if row.KVPath == "" {
		row.KVPath = "unknown"
	}
	if err := validateResolvedKVModes(row); err != nil {
		row.Status = statusFailed
		row.Success = boolPtr(false)
		row.Error = err.Error()
	}
	validation := runValidation(cfg, cell, row)
	row.ValidationKind = string(validation.Kind)
	row.ValidationStatus = string(validation.Status)
	row.ValidationExpected = validation.Expected
	row.ValidationObserved = validation.Observed
	row.ValidationError = validation.Error
	if validation.Kind == validationDecodeCorruptionGuard || validation.Kind == validationPromptFileRegression {
		row.ValidationCorruptionMarks = strings.Join(detectCorruptionMarkers(validation.Observed), ",")
	}
	if row.Status == statusOK && validation.Status == validationFailed {
		row.Status = statusFailed
		row.Success = boolPtr(false)
		if row.Error == "" {
			row.Error = validation.Error
		}
	}
	return enrichWithRunningInfo(ctx, cell.Host.Client, cfg.Model, row)
}

func enrichWithRunningInfo(ctx context.Context, client *api.Client, model string, row workerResult) workerResult {
	listCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	resp, err := client.ListRunning(listCtx)
	if err != nil {
		return row
	}
	for _, m := range resp.Models {
		if m.Name == model || m.Model == model || strings.HasPrefix(m.Name, model) || strings.HasPrefix(m.Model, model) {
			row.SizeBytes = m.Size
			row.SizeVRAMBytes = m.SizeVRAM
			row.ContextLength = m.ContextLength
			return row
		}
	}
	return row
}

func withOptionalTimeout(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func aggregateEpoch(rows []workerResult, wall time.Duration) epochAggregate {
	agg := epochAggregate{
		Host:                      rows[0].Host,
		HostLabel:                 rows[0].HostLabel,
		ServerVersion:             rows[0].ServerVersion,
		Model:                     rows[0].Model,
		ModelFamily:               rows[0].ModelFamily,
		ModelArch:                 rows[0].ModelArch,
		ModelSizeLabel:            rows[0].ModelSizeLabel,
		Quant:                     rows[0].Quant,
		KVModeRequested:           rows[0].KVModeRequested,
		KVModeRequestedK:          rows[0].KVModeRequestedK,
		KVModeRequestedV:          rows[0].KVModeRequestedV,
		RequestedCacheTypeK:       rows[0].RequestedCacheTypeK,
		RequestedCacheTypeV:       rows[0].RequestedCacheTypeV,
		RequestedMode:             rows[0].RequestedMode,
		EffectiveMode:             rows[0].EffectiveMode,
		KVBackendRequested:        rows[0].KVBackendRequested,
		KVAlgoResolved:            rows[0].KVAlgoResolved,
		KVAlgoResolvedK:           rows[0].KVAlgoResolvedK,
		KVAlgoResolvedV:           rows[0].KVAlgoResolvedV,
		KVPathK:                   rows[0].KVPathK,
		KVPathV:                   rows[0].KVPathV,
		KVSymmetric:               rows[0].KVSymmetric,
		KVAsymmetric:              rows[0].KVAsymmetric,
		SymmetricRequested:        rows[0].SymmetricRequested,
		SymmetricEffective:        rows[0].SymmetricEffective,
		FallbackApplied:           rows[0].FallbackApplied,
		KOnlyFallback:             rows[0].KOnlyFallback,
		FallbackReason:            rows[0].FallbackReason,
		TurboQuantPathKind:        rows[0].TurboQuantPathKind,
		PathKind:                  rows[0].PathKind,
		NativeTurboQuantActive:    rows[0].NativeTurboQuantActive,
		ReferenceTurboQuantActive: rows[0].ReferenceTurboQuantActive,
		FlashAttentionRequested:   rows[0].FlashAttentionRequested,
		FlashAttentionEffective:   rows[0].FlashAttentionEffective,
		FAEnabled:                 rows[0].FAEnabled,
		FARequiredForVTurbo:       rows[0].FARequiredForVTurbo,
		VTurboSupported:           rows[0].VTurboSupported,
		DetectedHeadDim:           rows[0].DetectedHeadDim,
		ArchitectureClass:         rows[0].ArchitectureClass,
		SupportTier:               rows[0].SupportTier,
		HybridKVArchitecture:      rows[0].HybridKVArchitecture,
		TQBlockSize:               rows[0].TQBlockSize,
		GPUStatsSource:            rows[0].GPUStatsSource,
		HostStatsSource:           rows[0].HostStatsSource,
		ValidationKind:            rows[0].ValidationKind,
		ValidationStatus:          rows[0].ValidationStatus,
		ValidationObserved:        rows[0].ValidationObserved,
		ValidationExpected:        rows[0].ValidationExpected,
		ValidationError:           rows[0].ValidationError,
		RequestedNumCtx:           rows[0].RequestedNumCtx,
		AttemptedNumCtx:           rows[0].AttemptedNumCtx,
		EffectiveNumCtx:           rows[0].EffectiveNumCtx,
		ContextRequested:          rows[0].ContextRequested,
		ContextEffective:          rows[0].ContextEffective,
		RequestedContextTopRung:   rows[0].RequestedContextTopRung,
		ContextLadderIndex:        rows[0].ContextLadderIndex,
		ContextFallbackReason:     rows[0].ContextFallbackReason,
		ContextFallbackDetail:     rows[0].ContextFallbackDetail,
		ContextFallbackStage:      rows[0].ContextFallbackStage,
		ContextFallbackClass:      rows[0].ContextFallbackClass,
		LadderRejectedRungs:       rows[0].LadderRejectedRungs,
		ModelFileSizeBytes:        rows[0].ModelFileSizeBytes,
		EstimatedKVFootprintBytes: rows[0].EstimatedKVFootprintBytes,
		KVBufferBytesEstimate:     rows[0].KVBufferBytesEstimate,
		VisibleGPUCount:           rows[0].VisibleGPUCount,
		PerGPUVRAMGiB:             rows[0].PerGPUVRAMGiB,
		TotalVisibleVRAMBytes:     rows[0].TotalVisibleVRAMBytes,
		ProcessVRAMBytes:          rows[0].ProcessVRAMBytes,
		GPUVRAMUsedBytes:          rows[0].GPUVRAMUsedBytes,
		GPUVRAMFreeBytes:          rows[0].GPUVRAMFreeBytes,
		PeakHostRAMDeltaBytes:     rows[0].PeakHostRAMDeltaBytes,
		HostRAMBeforeBytes:        rows[0].HostRAMBeforeBytes,
		HostRAMAfterLoadBytes:     rows[0].HostRAMAfterLoadBytes,
		HostRAMAfterPrefillBytes:  rows[0].HostRAMAfterPrefillBytes,
		HostRAMAfterDecodeBytes:   rows[0].HostRAMAfterDecodeBytes,
		UsedHostAssist:            rows[0].UsedHostAssist,
		UsedMMap:                  rows[0].UsedMMap,
		UsedCPUAssist:             rows[0].UsedCPUAssist,
		LongContextCapSource:      rows[0].LongContextCapSource,
		NativeContextAdvertised:   rows[0].NativeContextAdvertised,
		YarnContextAdvertised:     rows[0].YarnContextAdvertised,
		ValidationCorruptionMarks: rows[0].ValidationCorruptionMarks,
		FitStatus:                 rows[0].FitStatus,
		CorruptionStatus:          rows[0].CorruptionStatus,
		CorrectnessStatus:         rows[0].CorrectnessStatus,
		ResidencyKind:             rows[0].ResidencyKind,
		Notes:                     rows[0].Notes,
		Workload:                  rows[0].Workload,
		NumCtx:                    rows[0].NumCtx,
		PromptTokensTarget:        rows[0].PromptTokensTarget,
		MaxTokens:                 rows[0].MaxTokens,
		PromptTokens:              rows[0].PromptTokens,
		CtxXConc:                  rows[0].CtxXConc,
		Concurrency:               rows[0].Concurrency,
		Epoch:                     rows[0].Epoch,
		Warmup:                    rows[0].Warmup,
		PeakVRAMBytes:             rows[0].PeakVRAMBytes,
		AvgGPUUtil:                rows[0].AvgGPUUtil,
		PeakGPUUtil:               rows[0].PeakGPUUtil,
		GPUMetricsAvailable:       rows[0].GPUMetricsAvailable,
		HostRAMUsedBytes:          rows[0].HostRAMUsedBytes,
		PeakHostRAMBytes:          rows[0].PeakHostRAMBytes,
		HostMetricsAvailable:      rows[0].HostMetricsAvailable,
		FullGPUResidency:          rows[0].FullGPUResidency,
		GPUOffloadRegression:      rows[0].GPUOffloadRegression,
		ProcessorStateBefore:      rows[0].ProcessorStateBefore,
		ProcessorStateAfter:       rows[0].ProcessorStateAfter,
		Spilled:                   rows[0].Spilled,
		RunnerRSSBytes:            rows[0].RunnerRSSBytes,
		Status:                    statusOK,
		Success:                   boolPtr(true),
		WallMS:                    float64(wall) / float64(time.Millisecond),
		WallTimeS:                 float64(wall) / float64(time.Second),
	}
	var ttfts []float64
	var kvResolved []string
	var kvResolvedK []string
	var kvResolvedV []string
	var requestedModes []string
	var effectiveModes []string
	var kvAlgorithms []string
	var kvAlgorithmsK []string
	var kvAlgorithmsV []string
	var kvPaths []string
	var kvPathsK []string
	var kvPathsV []string
	var totalPromptEvalMS float64
	var totalEvalMS float64
	for _, row := range rows {
		agg.PromptEvalCount += row.PromptEvalCount
		agg.EvalCount += row.EvalCount
		agg.GeneratedTokens += row.GeneratedTokens
		agg.LiveKVTokensTotal += row.LiveKVTokensTotal
		agg.LoadMS += row.LoadMS
		agg.TotalMS += row.TotalMS
		totalPromptEvalMS += row.PromptEvalMS
		totalEvalMS += row.EvalMS
		ttfts = append(ttfts, row.TTFTMS)
		kvResolved = append(kvResolved, row.KVModeResolved)
		kvResolvedK = append(kvResolvedK, row.KVModeResolvedK)
		kvResolvedV = append(kvResolvedV, row.KVModeResolvedV)
		requestedModes = append(requestedModes, row.RequestedMode)
		effectiveModes = append(effectiveModes, row.EffectiveMode)
		kvAlgorithms = append(kvAlgorithms, row.KVAlgoResolved)
		kvAlgorithmsK = append(kvAlgorithmsK, row.KVAlgoResolvedK)
		kvAlgorithmsV = append(kvAlgorithmsV, row.KVAlgoResolvedV)
		kvPaths = append(kvPaths, row.KVPath)
		kvPathsK = append(kvPathsK, row.KVPathK)
		kvPathsV = append(kvPathsV, row.KVPathV)
		if ptrInt64Value(row.PeakVRAMBytes) > ptrInt64Value(agg.PeakVRAMBytes) {
			agg.PeakVRAMBytes = row.PeakVRAMBytes
		}
		if ptrInt64Value(row.PeakHostRAMBytes) > ptrInt64Value(agg.PeakHostRAMBytes) {
			agg.PeakHostRAMBytes = row.PeakHostRAMBytes
		}
		if ptrInt64Value(row.PeakHostRAMDeltaBytes) > ptrInt64Value(agg.PeakHostRAMDeltaBytes) {
			agg.PeakHostRAMDeltaBytes = row.PeakHostRAMDeltaBytes
		}
		if ptrInt64Value(row.TotalVisibleVRAMBytes) > ptrInt64Value(agg.TotalVisibleVRAMBytes) {
			agg.TotalVisibleVRAMBytes = row.TotalVisibleVRAMBytes
		}
		if ptrInt64Value(row.ProcessVRAMBytes) > ptrInt64Value(agg.ProcessVRAMBytes) {
			agg.ProcessVRAMBytes = row.ProcessVRAMBytes
		}
		if ptrInt64Value(row.HostRAMUsedBytes) > ptrInt64Value(agg.HostRAMUsedBytes) {
			agg.HostRAMUsedBytes = row.HostRAMUsedBytes
		}
		if ptrFloat64Value(row.PeakGPUUtil) > ptrFloat64Value(agg.PeakGPUUtil) {
			agg.PeakGPUUtil = row.PeakGPUUtil
		}
		if row.RunnerRSSBytes > agg.RunnerRSSBytes {
			agg.RunnerRSSBytes = row.RunnerRSSBytes
		}
		if agg.GPUStatsSource == "" {
			agg.GPUStatsSource = row.GPUStatsSource
		}
		if agg.HostStatsSource == "" {
			agg.HostStatsSource = row.HostStatsSource
		}
		if agg.ValidationKind == "" {
			agg.ValidationKind = row.ValidationKind
		}
		switch row.ValidationStatus {
		case string(validationFailed):
			agg.ValidationStatus = row.ValidationStatus
		case string(validationPassed):
			if agg.ValidationStatus == "" || agg.ValidationStatus == string(validationSkipped) || agg.ValidationStatus == string(validationScaffolded) {
				agg.ValidationStatus = row.ValidationStatus
			}
		case string(validationScaffolded):
			if agg.ValidationStatus == "" || agg.ValidationStatus == string(validationSkipped) {
				agg.ValidationStatus = row.ValidationStatus
			}
		case string(validationSkipped):
			if agg.ValidationStatus == "" {
				agg.ValidationStatus = row.ValidationStatus
			}
		}
		if agg.ValidationExpected == "" {
			agg.ValidationExpected = row.ValidationExpected
		}
		if agg.ValidationObserved == "" {
			agg.ValidationObserved = row.ValidationObserved
		}
		if agg.ValidationError == "" || row.ValidationStatus == string(validationFailed) {
			agg.ValidationError = row.ValidationError
		}
		agg.FullGPUResidency = agg.FullGPUResidency && row.FullGPUResidency
		agg.GPUOffloadRegression = agg.GPUOffloadRegression || row.GPUOffloadRegression
		agg.GPUMetricsAvailable = agg.GPUMetricsAvailable || row.GPUMetricsAvailable
		agg.HostMetricsAvailable = agg.HostMetricsAvailable || row.HostMetricsAvailable
		agg.Spilled = agg.Spilled || row.Spilled
		agg.UsedHostAssist = agg.UsedHostAssist || row.UsedHostAssist
		agg.UsedMMap = agg.UsedMMap || row.UsedMMap
		agg.UsedCPUAssist = agg.UsedCPUAssist || row.UsedCPUAssist
		if row.VisibleGPUCount > agg.VisibleGPUCount {
			agg.VisibleGPUCount = row.VisibleGPUCount
		}
		if agg.PerGPUVRAMGiB == "" {
			agg.PerGPUVRAMGiB = row.PerGPUVRAMGiB
		}
		if agg.LongContextCapSource == "" {
			agg.LongContextCapSource = row.LongContextCapSource
		}
		if row.NativeContextAdvertised > agg.NativeContextAdvertised {
			agg.NativeContextAdvertised = row.NativeContextAdvertised
		}
		if row.YarnContextAdvertised > agg.YarnContextAdvertised {
			agg.YarnContextAdvertised = row.YarnContextAdvertised
		}
		if agg.ValidationCorruptionMarks == "" {
			agg.ValidationCorruptionMarks = row.ValidationCorruptionMarks
		}
		if agg.LadderRejectedRungs == "" {
			agg.LadderRejectedRungs = row.LadderRejectedRungs
		}
		if agg.ContextFallbackReason == "" {
			agg.ContextFallbackReason = row.ContextFallbackReason
		}
		if agg.ContextFallbackDetail == "" {
			agg.ContextFallbackDetail = row.ContextFallbackDetail
		}
		if agg.ContextFallbackStage == "" {
			agg.ContextFallbackStage = row.ContextFallbackStage
		}
		if agg.ContextFallbackClass == "" {
			agg.ContextFallbackClass = row.ContextFallbackClass
		}
		if agg.ProcessorStateBefore == "" {
			agg.ProcessorStateBefore = row.ProcessorStateBefore
		}
		agg.ProcessorStateAfter = row.ProcessorStateAfter
		if row.Status != statusOK {
			agg.Status = row.Status
			agg.Success = row.Success
			if agg.Error == "" {
				agg.Error = row.Error
			}
		}
	}
	agg.KVModeResolved = uniqueOrMixed(kvResolved)
	agg.KVModeResolvedK = uniqueOrMixed(kvResolvedK)
	agg.KVModeResolvedV = uniqueOrMixed(kvResolvedV)
	agg.RequestedMode = uniqueOrMixed(requestedModes)
	agg.EffectiveMode = uniqueOrMixed(effectiveModes)
	agg.KVAlgoResolved = uniqueOrMixed(kvAlgorithms)
	agg.KVAlgoResolvedK = uniqueOrMixed(kvAlgorithmsK)
	agg.KVAlgoResolvedV = uniqueOrMixed(kvAlgorithmsV)
	agg.KVPath = uniqueOrMixed(kvPaths)
	agg.KVPathK = uniqueOrMixed(kvPathsK)
	agg.KVPathV = uniqueOrMixed(kvPathsV)
	if totalPromptEvalMS > 0 {
		agg.PrefillTPS = (float64(agg.PromptEvalCount) / totalPromptEvalMS) * 1000
	}
	if totalEvalMS > 0 {
		agg.DecodeTPS = (float64(agg.EvalCount) / totalEvalMS) * 1000
	}
	agg.LoadMS /= float64(len(rows))
	agg.TotalMS /= float64(len(rows))
	if agg.GPUMetricsAvailable {
		agg.AvgGPUUtil = rows[0].AvgGPUUtil
	}
	agg.TTFTMSMean = mean(ttfts)
	agg.TTFTMSP95 = percentile(ttfts, 0.95)
	agg.EffectiveCacheTypeK = firstNonEmpty(agg.KVModeResolvedK, agg.RequestedCacheTypeK)
	agg.EffectiveCacheTypeV = firstNonEmpty(agg.KVModeResolvedV, agg.RequestedCacheTypeV)
	agg.GPUVRAMUsedBytes = firstNonEmptyInt64(agg.PeakVRAMBytes, agg.GPUVRAMUsedBytes)
	agg.GPUVRAMFreeBytes = computeGPUVRAMFree(agg.TotalVisibleVRAMBytes, agg.GPUVRAMUsedBytes)
	return agg
}

func requestedKVBackend(kvMode string) string {
	if benchmarkModeUsesTurbo(kvMode) {
		return "cuda"
	}
	return ""
}

func uniqueOrMixed(values []string) string {
	filtered := make([]string, 0, len(values))
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			filtered = append(filtered, value)
		}
	}
	if len(filtered) == 0 {
		return "unknown"
	}
	first := filtered[0]
	for _, value := range filtered[1:] {
		if value != first {
			return "mixed"
		}
	}
	return first
}

func validateResolvedKVModes(row workerResult) error {
	requested := firstNonEmpty(row.KVModeRequested, "f16")
	requestedK, requestedV, _ := splitBenchmarkKVMode(requested)
	resolved := firstNonEmpty(row.KVModeResolved, "")
	resolvedK := firstNonEmpty(row.KVModeResolvedK, resolved)
	resolvedV := firstNonEmpty(row.KVModeResolvedV, resolved)
	if requestedK == "f16" && requestedV == "f16" {
		return nil
	}
	if isTurboQuantMode(row.KVModeRequestedV) && !strings.EqualFold(firstNonEmpty(row.KVModeRequestedV, "f16"), firstNonEmpty(resolvedV, "f16")) {
		if !row.FallbackApplied {
			return errors.New("requested V turboquant downgraded without fallback metadata")
		}
		if strings.TrimSpace(row.FallbackReason) == "" {
			return errors.New("requested V turboquant downgraded without fallback reason")
		}
		expectedKOnly := isTurboQuantMode(row.KVModeResolvedK) && !isTurboQuantMode(row.KVModeResolvedV)
		if row.KOnlyFallback != expectedKOnly {
			return fmt.Errorf("requested V turboquant fallback reported k_only_fallback=%t, want %t", row.KOnlyFallback, expectedKOnly)
		}
		if row.FARequiredForVTurbo && !row.FAEnabled && row.VTurboSupported {
			return errors.New("requested V turboquant reported supported despite Flash Attention gate")
		}
		return nil
	}
	if resolved == "" || (resolved == "f16" && resolvedK == "f16" && resolvedV == "f16") {
		return errors.New("requested kv mode did not resolve at runtime")
	}
	if !strings.EqualFold(requestedK, firstNonEmpty(resolvedK, "f16")) || !strings.EqualFold(requestedV, firstNonEmpty(resolvedV, "f16")) {
		if row.FallbackApplied {
			return nil
		}
		return fmt.Errorf("requested kv mode %s resolved as %s", requested, resolved)
	}
	if isTurboQuantMode(row.KVModeRequestedV) {
		if !row.VTurboSupported {
			return errors.New("requested V turboquant resolved without V-side support")
		}
		if row.FARequiredForVTurbo && !row.FAEnabled {
			return errors.New("requested V turboquant resolved without Flash Attention")
		}
	}
	if strings.HasPrefix(requested, "tq") && row.KVAlgoResolved != turboquant.AlgorithmPaper {
		return fmt.Errorf("requested kv mode %s resolved with algorithm %s", requested, firstNonEmpty(row.KVAlgoResolved, "unknown"))
	}
	if strings.TrimSpace(row.KVPath) == "" || row.KVPath == "unknown" {
		return errors.New("requested kv mode resolved without runtime path metadata")
	}
	if row.RequestedMode != "" && row.EffectiveMode != "" && row.RequestedMode != row.EffectiveMode && !row.FallbackApplied {
		return errors.New("requested and effective kv modes differ without fallback metadata")
	}
	return nil
}

func summarizeRequestedOrEffectiveMode(kType, vType string) string {
	kType = firstNonEmpty(strings.TrimSpace(kType), "f16")
	vType = firstNonEmpty(strings.TrimSpace(vType), "f16")
	if strings.EqualFold(kType, vType) {
		return kType
	}
	return fmt.Sprintf("k=%s,v=%s", kType, vType)
}

func isTurboQuantMode(mode string) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(mode)), "tq")
}

func kvAlgoForRequestedMode(kvMode string) string {
	if benchmarkModeUsesTurbo(kvMode) {
		return turboquant.AlgorithmPaper
	}
	return ""
}

func timedAggregateWithFullGPU(aggs []epochAggregate) bool {
	for _, agg := range aggs {
		if agg.Warmup {
			continue
		}
		if agg.Status != statusOK || !agg.FullGPUResidency {
			return false
		}
	}
	return true
}

func processorState(psOutput string) string {
	return inferGPUResidency(psOutput)
}

func spilledState(state string) bool {
	lower := strings.ToLower(strings.TrimSpace(state))
	return lower == "partial gpu"
}

func deriveResidencyKind(row workerResult) tqbenchschema.ResidencyKind {
	switch {
	case row.UsedCPUAssist:
		return tqbenchschema.ResidencyCPUAssist
	case row.UsedMMap:
		return tqbenchschema.ResidencyMMapAssist
	case row.UsedHostAssist || row.Spilled:
		return tqbenchschema.ResidencyMixed
	case row.FullGPUResidency:
		return tqbenchschema.ResidencyGPUOnly
	default:
		return tqbenchschema.ResidencyUnknown
	}
}

func computeGPUVRAMFree(total, used *int64) *int64 {
	if total == nil || used == nil {
		return nil
	}
	if *total < *used {
		return nil
	}
	free := *total - *used
	return &free
}

func deriveModelSizeLabel(model, override string) string {
	if strings.TrimSpace(override) != "" {
		return strings.TrimSpace(override)
	}
	lower := strings.ToLower(model)
	for _, marker := range []string{"397b", "32b", "30b", "27b", "9b", "7b", "1.7b"} {
		if strings.Contains(lower, marker) {
			return strings.ToUpper(marker)
		}
	}
	return "unknown"
}

func applyDerivedStatuses(row *workerResult) {
	row.PathKind = firstNonEmpty(row.PathKind, row.TurboQuantPathKind, row.KVPath, "unknown")
	row.ContextRequested = max(max(row.ContextRequested, row.RequestedNumCtx), row.NumCtx)
	row.ContextEffective = max(max(row.ContextEffective, row.EffectiveNumCtx), row.ContextRequested)
	row.RequestedCacheTypeK = firstNonEmpty(row.RequestedCacheTypeK, row.KVModeRequestedK)
	row.RequestedCacheTypeV = firstNonEmpty(row.RequestedCacheTypeV, row.KVModeRequestedV)
	row.EffectiveCacheTypeK = firstNonEmpty(row.EffectiveCacheTypeK, row.KVModeResolvedK, row.RequestedCacheTypeK, "f16")
	row.EffectiveCacheTypeV = firstNonEmpty(row.EffectiveCacheTypeV, row.KVModeResolvedV, row.RequestedCacheTypeV, "f16")
	row.SymmetricRequested = row.RequestedCacheTypeK == row.RequestedCacheTypeV
	row.SymmetricEffective = row.EffectiveCacheTypeK == row.EffectiveCacheTypeV
	row.FlashAttentionEffective = row.FAEnabled
	row.PromptTokens = max(max(row.PromptTokens, row.PromptEvalCount), row.PromptTokensTarget)
	row.KVBufferBytesEstimate = firstNonEmptyInt64(row.KVBufferBytesEstimate, row.EstimatedKVFootprintBytes)
	row.WallTimeS = row.WallMS / 1000

	switch row.Status {
	case statusUnsupported:
		row.FitStatus = string(tqbenchschema.FitStatusUnsupported)
		row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusUnsupported)
		row.CorruptionStatus = string(tqbenchschema.CorruptionStatusSkipped)
	case statusFailed:
		row.FitStatus = string(tqbenchschema.FitStatusFailed)
	default:
		if row.FallbackApplied {
			row.FitStatus = string(tqbenchschema.FitStatusFallback)
		} else {
			row.FitStatus = string(tqbenchschema.FitStatusFit)
		}
		if strings.TrimSpace(row.ValidationCorruptionMarks) != "" {
			row.CorruptionStatus = string(tqbenchschema.CorruptionStatusFail)
			row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusFail)
		} else {
			row.CorruptionStatus = string(tqbenchschema.CorruptionStatusPass)
			switch row.ValidationStatus {
			case string(validationFailed):
				row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusFail)
			case string(validationScaffolded):
				row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusScaffolded)
			case string(validationSkipped), "":
				row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusSkipped)
			default:
				row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusPass)
			}
		}
	}

	var notes []string
	if row.FallbackReason != "" {
		notes = append(notes, row.FallbackReason)
	}
	if row.ValidationError != "" {
		notes = append(notes, row.ValidationError)
	}
	if row.ResidencyKind != "" {
		notes = append(notes, "residency="+row.ResidencyKind)
	}
	row.Notes = strings.Join(notes, " | ")
}

func firstNonEmptyInt64(values ...*int64) *int64 {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func tokensPerSecond(count int, d time.Duration) float64 {
	if count <= 0 || d <= 0 {
		return 0
	}
	return float64(count) / d.Seconds()
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func percentile(values []float64, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	slices.Sort(sorted)
	if len(sorted) == 1 {
		return sorted[0]
	}
	pos := q * float64(len(sorted)-1)
	lower := int(math.Floor(pos))
	upper := int(math.Ceil(pos))
	if lower == upper {
		return sorted[lower]
	}
	weight := pos - float64(lower)
	return sorted[lower] + ((sorted[upper] - sorted[lower]) * weight)
}
