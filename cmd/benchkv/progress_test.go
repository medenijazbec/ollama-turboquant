package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestShouldEnableProgress(t *testing.T) {
	tests := []struct {
		name  string
		mode  progressMode
		isTTY bool
		want  bool
	}{
		{name: "auto tty", mode: progressAuto, isTTY: true, want: true},
		{name: "auto non tty", mode: progressAuto, isTTY: false, want: false},
		{name: "on", mode: progressOn, isTTY: false, want: true},
		{name: "off", mode: progressOff, isTTY: true, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := shouldEnableProgress(tt.mode, tt.isTTY); got != tt.want {
				t.Fatalf("shouldEnableProgress(%q, %t) = %t, want %t", tt.mode, tt.isTTY, got, tt.want)
			}
		})
	}
}

func TestEstimateTotalUnitsQuick(t *testing.T) {
	cfg := config{
		Hosts:       []hostTarget{{Label: "baseline"}},
		KVModes:     []string{"f16"},
		Workloads:   []workloadName{workloadPrefillHeavy, workloadDecodeGrowth, workloadParallelAmplifier},
		NumCtx:      []int{8192, 16384},
		Concurrency: []int{1, 2},
		Warmup:      1,
		Epochs:      3,
	}

	got := estimateTotalUnits(cfg)
	want := 49
	if got != want {
		t.Fatalf("estimateTotalUnits() = %d, want %d", got, want)
	}
}

func TestEstimateTotalUnitsStaircase(t *testing.T) {
	cfg := config{
		Hosts:       []hostTarget{{Label: "turbo"}},
		KVModes:     []string{"tq35"},
		Workloads:   []workloadName{workloadNearOOMStaircase},
		NumCtx:      []int{8192, 16384},
		Concurrency: []int{1, 2},
		Warmup:      1,
		Epochs:      2,
	}

	got := estimateTotalUnits(cfg)
	want := 13
	if got != want {
		t.Fatalf("estimateTotalUnits() = %d, want %d", got, want)
	}
}

func TestEstimateTotalUnitsWithPreflightSkipsUnsupportedStandardCells(t *testing.T) {
	cfg := config{
		Hosts:       []hostTarget{{BaseURL: "http://baseline", Label: "baseline"}},
		KVModes:     []string{"f16", "tq35"},
		Workloads:   []workloadName{workloadPrefillHeavy},
		NumCtx:      []int{8192},
		Concurrency: []int{1},
		Warmup:      0,
		Epochs:      1,
	}

	preflights := map[string]hostPreflight{
		"http://baseline": {
			KVSupport: map[string]kvSupportResult{
				"f16":  {Supported: true},
				"tq35": {Supported: false},
			},
		},
	}

	got := estimateTotalUnitsWithPreflight(cfg, preflights)
	want := 2
	if got != want {
		t.Fatalf("estimateTotalUnitsWithPreflight() = %d, want %d", got, want)
	}
}

func TestEstimateTotalUnitsWithPreflightSkipsUnsupportedStaircaseCells(t *testing.T) {
	cfg := config{
		Hosts:       []hostTarget{{BaseURL: "http://baseline", Label: "baseline"}},
		KVModes:     []string{"f16", "tq35"},
		Workloads:   []workloadName{workloadNearOOMStaircase},
		NumCtx:      []int{8192},
		Concurrency: []int{1},
		Warmup:      0,
		Epochs:      1,
	}

	preflights := map[string]hostPreflight{
		"http://baseline": {
			KVSupport: map[string]kvSupportResult{
				"f16":  {Supported: true},
				"tq35": {Supported: false},
			},
		},
	}

	got := estimateTotalUnitsWithPreflight(cfg, preflights)
	want := 2
	if got != want {
		t.Fatalf("estimateTotalUnitsWithPreflight() = %d, want %d", got, want)
	}
}

func TestProgressTrackerRender(t *testing.T) {
	var buf bytes.Buffer
	tracker := newTestProgressTracker(&buf, true, 10, 10, false)

	tracker.SetCurrent(progressStep{
		Phase:       "epoch",
		HostLabel:   "baseline",
		KVMode:      "tq35",
		Workload:    "prefill-heavy",
		NumCtx:      16384,
		Concurrency: 2,
		Epoch:       2,
		EpochTotal:  3,
	})
	tracker.Advance(4)

	out := buf.String()
	for _, part := range []string{"tick |", "baseline", "tq35", "prefill-heavy", "ctx=16384", "conc=2", "epoch 2/3"} {
		if !strings.Contains(out, part) {
			t.Fatalf("rendered progress missing %q in %q", part, out)
		}
	}
	if !strings.Contains(out, "\r") {
		t.Fatalf("expected carriage return rendering, got %q", out)
	}
}

func TestProgressTrackerFinishPrintsOneNewline(t *testing.T) {
	var buf bytes.Buffer
	tracker := newTestProgressTracker(&buf, true, 10, 10, false)

	tracker.Finish()
	tracker.Finish()

	if got := strings.Count(buf.String(), "\n"); got != 1 {
		t.Fatalf("newline count = %d, want 1", got)
	}
}

func TestRemainingStaircaseUnits(t *testing.T) {
	cfg := config{
		NumCtx:      []int{8192, 16384, 32768},
		Concurrency: []int{1, 2},
		Warmup:      1,
		Epochs:      2,
	}

	got := remainingStaircaseUnits(cfg, 1, 16384)
	want := 12
	if got != want {
		t.Fatalf("remainingStaircaseUnits() = %d, want %d", got, want)
	}
}

func TestProgressTrackerPulseAdvancesLine(t *testing.T) {
	var buf bytes.Buffer
	tracker := newTestProgressTracker(&buf, true, 10, 10, false)

	tracker.SetCurrent(progressStep{
		Phase:      "preflight",
		HostLabel:  "baseline",
		Epoch:      1,
		EpochTotal: 1,
	})
	first := tracker.line()
	tracker.pulseIndex++
	second := tracker.line()
	if first == second {
		t.Fatalf("expected pulse marker to change line, got %q", first)
	}
}

func TestProgressTrackerSetTotalClampsCompleted(t *testing.T) {
	var buf bytes.Buffer
	tracker := newTestProgressTracker(&buf, true, 10, 10, false)

	tracker.Advance(7)
	tracker.SetTotal(5)

	if tracker.total != 5 {
		t.Fatalf("total = %d, want 5", tracker.total)
	}
	if tracker.completed != 5 {
		t.Fatalf("completed = %d, want 5", tracker.completed)
	}
	if !strings.Contains(buf.String(), "100%") {
		t.Fatalf("expected updated render to include 100%%, got %q", buf.String())
	}
}
