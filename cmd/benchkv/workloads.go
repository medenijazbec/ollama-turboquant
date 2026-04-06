package main

import "cmp"

func effectiveFAModes(cfg config) []bool {
	if len(cfg.FAModes) == 0 {
		return []bool{false}
	}
	return cfg.FAModes
}

func effectiveQJLModes(modes []bool) []bool {
	if len(modes) == 0 {
		return []bool{false}
	}
	return modes
}

func effectiveResidualTailTokens(cfg config) []int {
	if cfg.ResidualTailTokens > 0 {
		return []int{cfg.ResidualTailTokens}
	}
	if cfg.Profile == "large-context" || cfg.Profile == "test-matrix" || cfg.Profile == "memory" {
		return []int{0, 128}
	}
	for _, workload := range cfg.Workloads {
		if workload == workloadFitCeiling || workload == workloadLongContextRecall || workload == workloadNIAHRetrieval {
			return []int{0, 128}
		}
	}
	return []int{0}
}

func buildStandardCells(cfg config) []sweepCell {
	switch cfg.Profile {
	case "impact":
		return buildImpactCells(cfg)
	case "regression":
		return buildRegressionCells(cfg)
	case "turbo-benefit":
		return buildTurboBenefitCells(cfg)
	case "large-context":
		return buildLargeContextCells(cfg)
	case "capacity", "spill", "staircase":
		return nil
	}

	var cells []sweepCell
	for _, host := range cfg.Hosts {
		for _, kvMode := range cfg.KVModes {
			for _, faRequested := range effectiveFAModes(cfg) {
				for _, qjlKRequested := range effectiveQJLModes(cfg.QJLKModes) {
					for _, qjlVRequested := range effectiveQJLModes(cfg.QJLVModes) {
						for _, residualTailTokens := range effectiveResidualTailTokens(cfg) {
							for _, workload := range cfg.Workloads {
								if workload == workloadNearOOMStaircase {
									continue
								}
								for _, spec := range expandWorkload(cfg, workload) {
									cells = append(cells, sweepCell{
										Host:               host,
										KVMode:             kvMode,
										Workload:           spec,
										FARequested:        faRequested,
										QJLKRequested:      qjlKRequested,
										QJLVRequested:      qjlVRequested,
										ResidualTailTokens: residualTailTokens,
									})
								}
							}
						}
					}
				}
			}
		}
	}
	return cells
}

func buildRegressionCells(cfg config) []sweepCell {
	var cells []sweepCell
	for _, host := range cfg.Hosts {
		for _, kvMode := range cfg.KVModes {
			for _, faRequested := range effectiveFAModes(cfg) {
				for _, qjlKRequested := range effectiveQJLModes(cfg.QJLKModes) {
					for _, qjlVRequested := range effectiveQJLModes(cfg.QJLVModes) {
						for _, residualTailTokens := range effectiveResidualTailTokens(cfg) {
							for _, workload := range cfg.Workloads {
								if workload == workloadNearOOMStaircase {
									continue
								}
								for _, spec := range expandWorkload(cfg, workload) {
									cells = append(cells, sweepCell{
										Host:               host,
										KVMode:             kvMode,
										Workload:           spec,
										FARequested:        faRequested,
										QJLKRequested:      qjlKRequested,
										QJLVRequested:      qjlVRequested,
										ResidualTailTokens: residualTailTokens,
									})
								}
							}
						}
					}
				}
			}
		}
	}
	return cells
}

func buildTurboBenefitCells(cfg config) []sweepCell {
	var cells []sweepCell
	for _, host := range cfg.Hosts {
		if host.KVSupportMode != hostKVSupportRequest {
			continue
		}
		for _, kvMode := range cfg.KVModes {
			for _, faRequested := range effectiveFAModes(cfg) {
				for _, qjlKRequested := range effectiveQJLModes(cfg.QJLKModes) {
					for _, qjlVRequested := range effectiveQJLModes(cfg.QJLVModes) {
						for _, residualTailTokens := range effectiveResidualTailTokens(cfg) {
							for _, workload := range cfg.Workloads {
								if workload == workloadNearOOMStaircase {
									continue
								}
								for _, spec := range expandWorkload(cfg, workload) {
									cells = append(cells, sweepCell{
										Host:               host,
										KVMode:             kvMode,
										Workload:           spec,
										FARequested:        faRequested,
										QJLKRequested:      qjlKRequested,
										QJLVRequested:      qjlVRequested,
										ResidualTailTokens: residualTailTokens,
									})
								}
							}
						}
					}
				}
			}
		}
	}
	return cells
}

func buildImpactCells(cfg config) []sweepCell {
	specs := []workloadSpec{
		{
			Name:               workloadPrefillHeavy,
			NumCtx:             16384,
			PromptTokensTarget: cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(16384)),
			MaxTokens:          cmp.Or(cfg.MaxTokens, 128),
			Concurrency:        2,
		},
		{
			Name:               workloadDecodeGrowth,
			NumCtx:             8192,
			PromptTokensTarget: cmp.Or(cfg.PromptTokens, 2048),
			MaxTokens:          cmp.Or(cfg.MaxTokens, 768),
			Concurrency:        1,
		},
	}

	var cells []sweepCell
	for _, host := range cfg.Hosts {
		for _, kvMode := range cfg.KVModes {
			for _, faRequested := range effectiveFAModes(cfg) {
				for _, qjlKRequested := range effectiveQJLModes(cfg.QJLKModes) {
					for _, qjlVRequested := range effectiveQJLModes(cfg.QJLVModes) {
						for _, residualTailTokens := range effectiveResidualTailTokens(cfg) {
							for _, spec := range specs {
								cells = append(cells, sweepCell{
									Host:               host,
									KVMode:             kvMode,
									Workload:           spec,
									FARequested:        faRequested,
									QJLKRequested:      qjlKRequested,
									QJLVRequested:      qjlVRequested,
									ResidualTailTokens: residualTailTokens,
								})
							}
						}
					}
				}
			}
		}
	}
	return cells
}

func buildLargeContextCells(cfg config) []sweepCell {
	var cells []sweepCell
	for _, host := range cfg.Hosts {
		for _, kvMode := range cfg.KVModes {
			for _, faRequested := range effectiveFAModes(cfg) {
				for _, qjlKRequested := range effectiveQJLModes(cfg.QJLKModes) {
					for _, qjlVRequested := range effectiveQJLModes(cfg.QJLVModes) {
						for _, residualTailTokens := range effectiveResidualTailTokens(cfg) {
							for _, workload := range cfg.Workloads {
								spec, ok := makeWorkloadSpec(cfg, workload, cfg.StretchContext, 1)
								if !ok {
									continue
								}
								cells = append(cells, sweepCell{
									Host:               host,
									KVMode:             kvMode,
									Workload:           spec,
									FARequested:        faRequested,
									QJLKRequested:      qjlKRequested,
									QJLVRequested:      qjlVRequested,
									ResidualTailTokens: residualTailTokens,
								})
							}
						}
					}
				}
			}
		}
	}
	return cells
}

func expandWorkload(cfg config, workload workloadName) []workloadSpec {
	var specs []workloadSpec
	for _, numCtx := range cfg.NumCtx {
		for _, concurrency := range cfg.Concurrency {
			spec, ok := makeWorkloadSpec(cfg, workload, numCtx, concurrency)
			if ok {
				specs = append(specs, spec)
			}
		}
	}
	return specs
}

func makeWorkloadSpec(cfg config, workload workloadName, numCtx, concurrency int) (workloadSpec, bool) {
	switch workload {
	case workloadPrefillHeavy:
		promptTokens := cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(numCtx))
		maxTokens := cmp.Or(cfg.MaxTokens, 128)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: concurrency}, true
	case workloadDecodeGrowth:
		promptTokens := cmp.Or(cfg.PromptTokens, 2048)
		maxTokens := cfg.MaxTokens
		if maxTokens == 0 {
			maxTokens = min(4096, numCtx-2304)
		}
		if maxTokens <= 0 {
			return workloadSpec{}, false
		}
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: concurrency}, true
	case workloadParallelAmplifier:
		if concurrency != 1 && concurrency != 2 && concurrency != 4 {
			return workloadSpec{}, false
		}
		promptTokens := cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(numCtx))
		maxTokens := cmp.Or(cfg.MaxTokens, 128)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: concurrency}, true
	case workloadNearOOMStaircase:
		promptTokens := cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(numCtx))
		maxTokens := cmp.Or(cfg.MaxTokens, 128)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: concurrency}, true
	case workloadFitCeiling:
		promptTokens := cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(numCtx))
		maxTokens := max(cmp.Or(cfg.MaxTokens, 32), cfg.MinFitDecode)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: 1}, true
	case workloadLongContextRecall:
		promptTokens := cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(numCtx))
		maxTokens := cmp.Or(cfg.MaxTokens, 64)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: 1}, true
	case workloadNIAHRetrieval:
		promptTokens := cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(numCtx))
		maxTokens := cmp.Or(cfg.MaxTokens, 32)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: 1}, true
	case workloadLongJSONRetention:
		promptTokens := cmp.Or(cfg.PromptTokens, promptTokensAt90Percent(numCtx))
		maxTokens := cmp.Or(cfg.MaxTokens, 96)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: 1}, true
	case workloadPromptFileRegress:
		promptTokens := cmp.Or(cfg.PromptTokens, max(320, min(numCtx/2, promptTokensAt90Percent(numCtx))))
		maxTokens := cmp.Or(cfg.MaxTokens, 96)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: 1}, true
	case workloadDecodeCorruption:
		promptTokens := cmp.Or(cfg.PromptTokens, max(512, min(numCtx/2, promptTokensAt90Percent(numCtx))))
		maxTokens := cmp.Or(cfg.MaxTokens, 2048)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: 1}, true
	case workloadAgenticStructured:
		promptTokens := cmp.Or(cfg.PromptTokens, max(1024, min(numCtx/2, promptTokensAt90Percent(numCtx))))
		maxTokens := cmp.Or(cfg.MaxTokens, 256)
		return workloadSpec{Name: workload, NumCtx: numCtx, PromptTokensTarget: promptTokens, MaxTokens: maxTokens, Concurrency: 1}, true
	default:
		return workloadSpec{}, false
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func promptTokensAt90Percent(numCtx int) int {
	return (numCtx * 9) / 10
}
