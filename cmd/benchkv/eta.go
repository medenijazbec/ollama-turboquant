package main

import "time"

func estimateStageETA(stage string, promptDone, promptTarget, genDone, genTarget int, rollingPromptTPS, rollingDecodeTPS float64, stageElapsed time.Duration) (*time.Duration, string, bool) {
	switch stage {
	case "prefill":
		if promptTarget > 0 && promptDone > 0 && promptDone < promptTarget && rollingPromptTPS > 0 {
			remaining := float64(promptTarget-promptDone) / rollingPromptTPS
			eta := time.Duration(remaining * float64(time.Second))
			return &eta, "medium", true
		}
	case "decode":
		if genTarget > 0 && genDone >= 0 && genDone < genTarget && rollingDecodeTPS > 0 {
			remaining := float64(genTarget-genDone) / rollingDecodeTPS
			eta := time.Duration(remaining * float64(time.Second))
			return &eta, "high", true
		}
	case "validate":
		eta := maxStageDuration(2*time.Second, 5*time.Second-stageElapsed)
		return &eta, "medium", false
	case "save":
		eta := maxStageDuration(500*time.Millisecond, 2*time.Second-stageElapsed)
		return &eta, "medium", false
	case "loading", "queued":
		return nil, "low", false
	}
	return nil, "low", false
}

func maxStageDuration(a, b time.Duration) time.Duration {
	if a > b {
		return a
	}
	return b
}
