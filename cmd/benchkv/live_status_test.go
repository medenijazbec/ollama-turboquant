package main

import (
	"strings"
	"testing"
)

func TestRenderLiveSampleLineIncludesModesAndFallback(t *testing.T) {
	line := renderLiveSampleLine(liveSample{
		Suite:                   "ab-minimal",
		TestIndex:               3,
		TestTotal:               16,
		HostLabel:               "turbo",
		Model:                   "qwen3-1.7b:udq4kxl",
		Workload:                "long-context-recall",
		Stage:                   "decode",
		RequestedContext:        16384,
		EffectiveContext:        16384,
		RequestedKVMode:         "q8_0/tq35",
		EffectiveKVMode:         "q8_0/tq35",
		FlashAttentionRequested: true,
		FlashAttentionEffective: true,
		ElapsedSec:              2,
		ProgressKnown:           true,
		ProgressPercent:         50,
		Epoch:                   1,
		EpochTotal:              1,
		LadderPosition:          2,
		LadderTotal:             6,
		CumulativePromptTPS:     100,
		CumulativeDecodeTPS:     20,
		GeneratedTokens:         10,
		PromptTokensProcessed:   20,
		FallbackApplied:         true,
		ValidationState:         "running",
	})
	if !strings.Contains(line, "req=q8_0/tq35 eff=q8_0/tq35") {
		t.Fatalf("line missing requested/effective kv: %q", line)
	}
	if !strings.Contains(line, "fallback=yes") {
		t.Fatalf("line missing fallback state: %q", line)
	}
}

func TestRenderLiveGPUCompactStableOrder(t *testing.T) {
	got := renderLiveGPUCompact(liveSample{
		GPUVRAMUsedBytesByGPU: map[string]int64{"1": 2 << 30, "0": 1 << 30},
		GPUVRAMFreeBytesByGPU: map[string]int64{"1": 2 << 30, "0": 3 << 30},
		GPUUtilPercentByGPU:   map[string]float64{"1": 90, "0": 80},
	})
	if !strings.HasPrefix(got, "gpu0=") {
		t.Fatalf("expected gpu0 first, got %q", got)
	}
	if !strings.Contains(got, "gpu1=") {
		t.Fatalf("expected gpu1 in output, got %q", got)
	}
}
