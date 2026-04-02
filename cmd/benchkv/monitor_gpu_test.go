package main

import "testing"

func TestParseNvidiaSMI(t *testing.T) {
	out := "1000, 50\n2000, 70\n"
	peakVRAM, meanUtil, ok := parseNvidiaSMI(out)
	if !ok {
		t.Fatal("expected parse to succeed")
	}
	if peakVRAM != int64(3000*1024*1024) {
		t.Fatalf("unexpected vram bytes: %d", peakVRAM)
	}
	if meanUtil != 60 {
		t.Fatalf("unexpected mean util: %.2f", meanUtil)
	}
}

func TestGPUMonitorStatsReportUnavailableSource(t *testing.T) {
	monitor := newGPUMonitor(0, true)
	stats := monitor.stats()
	if stats.Available {
		t.Fatalf("expected unavailable stats, got %+v", stats)
	}
	if stats.Source != "unavailable" {
		t.Fatalf("Source = %q, want unavailable", stats.Source)
	}
	if stats.PeakVRAMBytes != nil {
		t.Fatalf("expected nil VRAM bytes, got %+v", stats)
	}
}

func TestGPUMonitorStatsKeepSourceAndNilProcessVRAM(t *testing.T) {
	monitor := newGPUMonitor(0, true)
	monitor.available = true
	monitor.source = "nvidia-smi"
	monitor.peakVRAM = 1024
	monitor.sumMeanUtil = 50
	monitor.peakMeanUtil = 75
	monitor.samples = 2

	stats := monitor.stats()
	if !stats.Available || stats.Source != "nvidia-smi" {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if stats.ProcessVRAMBytes != nil {
		t.Fatalf("expected nil process VRAM, got %+v", stats)
	}
}
