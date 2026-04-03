package main

import "testing"

func TestParseNvidiaSMI(t *testing.T) {
	out := "0, 1000, 2000, 3000, 50\n1, 2000, 4000, 6000, 70\n"
	peakVRAM, meanUtil, totalVisible, perGPUUsed, perGPUFree, visibleCount, ok := parseNvidiaSMI(out)
	if !ok {
		t.Fatal("expected parse to succeed")
	}
	if peakVRAM != int64(3000*1024*1024) {
		t.Fatalf("unexpected vram bytes: %d", peakVRAM)
	}
	if meanUtil != 60 {
		t.Fatalf("unexpected mean util: %.2f", meanUtil)
	}
	if totalVisible == nil || *totalVisible != int64(9000*1024*1024) {
		t.Fatalf("unexpected total visible VRAM: %+v", totalVisible)
	}
	if visibleCount != 2 {
		t.Fatalf("visibleCount = %d, want 2", visibleCount)
	}
	if perGPUUsed["0"] != int64(1000*1024*1024) || perGPUFree["1"] != int64(4000*1024*1024) {
		t.Fatalf("unexpected per-GPU maps: used=%v free=%v", perGPUUsed, perGPUFree)
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

func TestGPUMonitorStatsCarryPerGPUAndVisibleTotals(t *testing.T) {
	monitor := newGPUMonitor(0, true)
	monitor.available = true
	monitor.source = "nvml"
	monitor.peakVRAM = 1024
	monitor.totalVisibleVRAM = 4096
	monitor.hasTotalVisibleVRAM = true
	monitor.perGPUUsed = map[string]int64{"0": 1024, "1": 2048}
	monitor.perGPUFree = map[string]int64{"0": 3072, "1": 2048}
	monitor.visibleGPUCount = 2
	monitor.samples = 1

	stats := monitor.stats()
	if stats.TotalVisibleVRAMBytes == nil || *stats.TotalVisibleVRAMBytes != 4096 {
		t.Fatalf("unexpected total visible VRAM: %+v", stats)
	}
	if stats.VisibleGPUCount != 2 {
		t.Fatalf("VisibleGPUCount = %d, want 2", stats.VisibleGPUCount)
	}
	if stats.PerGPUUsedBytes["1"] != 2048 || stats.PerGPUFreeBytes["0"] != 3072 {
		t.Fatalf("unexpected per-GPU maps: %+v", stats)
	}
}
