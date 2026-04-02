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
	row := workerResult{
		Host:               cell.Host.BaseURL,
		HostLabel:          cell.Host.Label,
		ServerVersion:      preflight.Version,
		Model:              cfg.Model,
		Quant:              preflight.ModelQuant,
		KVModeRequested:    cell.KVMode,
		KVModeRequestedK:   cell.KVMode,
		KVModeRequestedV:   cell.KVMode,
		KVBackendRequested: requestedKVBackend(cell.KVMode),
		KVAlgoResolved:     firstNonEmpty(kvAlgoForRequestedMode(cell.KVMode), "unknown"),
		KVPath:             "unknown",
		Workload:           string(cell.Workload.Name),
		NumCtx:             cell.Workload.NumCtx,
		PromptTokensTarget: cell.Workload.PromptTokensTarget,
		MaxTokens:          cell.Workload.MaxTokens,
		CtxXConc:           cell.Workload.NumCtx * cell.Workload.Concurrency,
		Concurrency:        cell.Workload.Concurrency,
		Epoch:              0,
		Warmup:             false,
		RunnerRSSBytes:     -1,
		GPUResidency:       "unknown",
		Status:             statusUnsupported,
		Success:            nil,
		Error:              errText,
		RecordedAt:         time.Now().UTC(),
	}
	agg := epochAggregate{
		Host:               row.Host,
		HostLabel:          row.HostLabel,
		ServerVersion:      row.ServerVersion,
		Model:              row.Model,
		Quant:              row.Quant,
		KVModeRequested:    row.KVModeRequested,
		KVModeRequestedK:   row.KVModeRequestedK,
		KVModeRequestedV:   row.KVModeRequestedV,
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
			results[idx] = runWorker(cfg, cell, cal, epoch, warmup, idx)
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
		results[i].PeakVRAMBytes = stats.PeakVRAMBytes
		results[i].AvgGPUUtil = stats.AvgGPUUtil
		results[i].PeakGPUUtil = stats.PeakGPUUtil
		results[i].HostMetricsAvailable = hostStats.Available
		results[i].HostRAMUsedBytes = hostStats.HostRAMUsedBytes
		results[i].PeakHostRAMBytes = hostStats.PeakHostRAMBytes
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
	}

	return results, aggregateEpoch(results, time.Since(start))
}

func runWorker(cfg config, cell sweepCell, cal promptCalibration, epoch int, warmup bool, workerIndex int) workerResult {
	row := workerResult{
		Host:               cell.Host.BaseURL,
		HostLabel:          cell.Host.Label,
		Model:              cfg.Model,
		KVModeRequested:    cell.KVMode,
		KVModeRequestedK:   cell.KVMode,
		KVModeRequestedV:   cell.KVMode,
		KVBackendRequested: requestedKVBackend(cell.KVMode),
		Workload:           string(cell.Workload.Name),
		NumCtx:             cell.Workload.NumCtx,
		PromptTokensTarget: cell.Workload.PromptTokensTarget,
		MaxTokens:          cell.Workload.MaxTokens,
		CtxXConc:           cell.Workload.NumCtx * cell.Workload.Concurrency,
		Concurrency:        cell.Workload.Concurrency,
		WorkerIndex:        workerIndex,
		Epoch:              epoch,
		Warmup:             warmup,
		RunnerRSSBytes:     -1,
		GPUResidency:       "unknown",
		Status:             statusFailed,
		RecordedAt:         time.Now().UTC(),
	}

	prompt := renderWorkerPrompt(cal, epoch, workerIndex)
	stream := cfg.Stream
	keepAlive := api.Duration{Duration: cfg.KeepAlive}
	options, disposition := buildGenerateOptions(cell.Host, cell.KVMode, cell.Workload.NumCtx, cell.Workload.MaxTokens, cfg.Seed, cfg.Temperature)
	if !disposition.Supported {
		row.Status = statusUnsupported
		row.Success = nil
		row.Error = disposition.Error
		return row
	}

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
	row.KVModeResolved = firstNonEmpty(finalMetrics.ResolvedKVCacheType, finalMetrics.KVCacheEffective, row.KVModeRequested)
	row.KVModeResolvedK = firstNonEmpty(finalMetrics.ResolvedKVCacheTypeK, row.KVModeRequestedK)
	row.KVModeResolvedV = firstNonEmpty(finalMetrics.ResolvedKVCacheTypeV, row.KVModeRequestedV)
	row.KVAlgoResolved = firstNonEmpty(finalMetrics.KVAlgoResolved, kvAlgoForRequestedMode(row.KVModeResolved))
	row.KVAlgoResolvedK = firstNonEmpty(finalMetrics.KVAlgoResolvedK, row.KVAlgoResolved)
	row.KVAlgoResolvedV = firstNonEmpty(finalMetrics.KVAlgoResolvedV, row.KVAlgoResolved)
	row.KVPath = firstNonEmpty(finalMetrics.KVCachePath, "unknown")
	row.KVPathK = firstNonEmpty(finalMetrics.KVCachePathK, row.KVPath)
	row.KVPathV = firstNonEmpty(finalMetrics.KVCachePathV, row.KVPath)
	row.KVSymmetric = finalMetrics.KVSymmetric
	row.KVAsymmetric = finalMetrics.KVAsymmetric
	row.FallbackReason = finalMetrics.FallbackReason
	row.TurboQuantPathKind = finalMetrics.TurboQuantPathKind
	row.NativeTurboQuantActive = finalMetrics.NativeTurboQuantActive
	row.ReferenceTurboQuantActive = finalMetrics.ReferenceTurboQuantActive
	row.FAEnabled = finalMetrics.FAEnabled
	row.VTurboSupported = finalMetrics.VTurboSupported
	row.TQBlockSize = finalMetrics.TQBlockSize
	row.PromptEvalCount = finalMetrics.PromptEvalCount
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
	if row.KVPath == "" {
		row.KVPath = "unknown"
	}
	if err := validateResolvedKVMode(row.KVModeRequested, row.KVModeResolved, row.KVAlgoResolved, row.KVPath); err != nil {
		row.Status = statusFailed
		row.Success = boolPtr(false)
		row.Error = err.Error()
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
		Quant:                     rows[0].Quant,
		KVModeRequested:           rows[0].KVModeRequested,
		KVModeRequestedK:          rows[0].KVModeRequestedK,
		KVModeRequestedV:          rows[0].KVModeRequestedV,
		KVBackendRequested:        rows[0].KVBackendRequested,
		KVAlgoResolved:            rows[0].KVAlgoResolved,
		KVAlgoResolvedK:           rows[0].KVAlgoResolvedK,
		KVAlgoResolvedV:           rows[0].KVAlgoResolvedV,
		KVPathK:                   rows[0].KVPathK,
		KVPathV:                   rows[0].KVPathV,
		KVSymmetric:               rows[0].KVSymmetric,
		KVAsymmetric:              rows[0].KVAsymmetric,
		FallbackReason:            rows[0].FallbackReason,
		TurboQuantPathKind:        rows[0].TurboQuantPathKind,
		NativeTurboQuantActive:    rows[0].NativeTurboQuantActive,
		ReferenceTurboQuantActive: rows[0].ReferenceTurboQuantActive,
		FAEnabled:                 rows[0].FAEnabled,
		VTurboSupported:           rows[0].VTurboSupported,
		TQBlockSize:               rows[0].TQBlockSize,
		Workload:                  rows[0].Workload,
		NumCtx:                    rows[0].NumCtx,
		PromptTokensTarget:        rows[0].PromptTokensTarget,
		MaxTokens:                 rows[0].MaxTokens,
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
	}
	var ttfts []float64
	var kvResolved []string
	var kvResolvedK []string
	var kvResolvedV []string
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
		if ptrInt64Value(row.HostRAMUsedBytes) > ptrInt64Value(agg.HostRAMUsedBytes) {
			agg.HostRAMUsedBytes = row.HostRAMUsedBytes
		}
		if ptrFloat64Value(row.PeakGPUUtil) > ptrFloat64Value(agg.PeakGPUUtil) {
			agg.PeakGPUUtil = row.PeakGPUUtil
		}
		if row.RunnerRSSBytes > agg.RunnerRSSBytes {
			agg.RunnerRSSBytes = row.RunnerRSSBytes
		}
		agg.FullGPUResidency = agg.FullGPUResidency && row.FullGPUResidency
		agg.GPUOffloadRegression = agg.GPUOffloadRegression || row.GPUOffloadRegression
		agg.GPUMetricsAvailable = agg.GPUMetricsAvailable || row.GPUMetricsAvailable
		agg.HostMetricsAvailable = agg.HostMetricsAvailable || row.HostMetricsAvailable
		agg.Spilled = agg.Spilled || row.Spilled
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
	return agg
}

func requestedKVBackend(kvMode string) string {
	if strings.HasPrefix(kvMode, "tq") {
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

func validateResolvedKVMode(requested, resolved, algo, path string) error {
	requested = firstNonEmpty(requested, "f16")
	resolved = firstNonEmpty(resolved, "")
	if requested == "f16" {
		return nil
	}
	if resolved == "" || resolved == "f16" {
		return errors.New("requested kv mode did not resolve at runtime")
	}
	if resolved != requested {
		return fmt.Errorf("requested kv mode %s resolved as %s", requested, resolved)
	}
	if strings.HasPrefix(requested, "tq") && algo != turboquant.AlgorithmPaper {
		return fmt.Errorf("requested kv mode %s resolved with algorithm %s", requested, firstNonEmpty(algo, "unknown"))
	}
	if strings.TrimSpace(path) == "" || path == "unknown" {
		return errors.New("requested kv mode resolved without runtime path metadata")
	}
	return nil
}

func kvAlgoForRequestedMode(kvMode string) string {
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(kvMode)), "tq") {
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
