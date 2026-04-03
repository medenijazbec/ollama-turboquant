package main

import (
	"context"
	"maps"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

type gpuPoller interface {
	poll() (gpuSample, bool)
	close()
	source() string
}

type gpuSample struct {
	peakVRAMBytes         int64
	meanUtil              float64
	processVRAMBytes      *int64
	totalVisibleVRAMBytes *int64
	perGPUUsedBytes       map[string]int64
	perGPUFreeBytes       map[string]int64
	visibleGPUCount       int
}

type gpuMonitor struct {
	interval time.Duration
	enabled  bool

	initOnce sync.Once
	poller   gpuPoller

	mu                  sync.Mutex
	available           bool
	source              string
	peakVRAM            int64
	sumMeanUtil         float64
	peakMeanUtil        float64
	processVRAM         int64
	hasProcessVRAM      bool
	totalVisibleVRAM    int64
	hasTotalVisibleVRAM bool
	perGPUUsed          map[string]int64
	perGPUFree          map[string]int64
	visibleGPUCount     int
	samples             int
}

func newGPUMonitor(interval time.Duration, enabled bool) *gpuMonitor {
	return &gpuMonitor{
		interval: interval,
		enabled:  enabled,
	}
}

func (m *gpuMonitor) run(ctx context.Context) {
	if !m.enabled {
		return
	}
	if m.interval <= 0 {
		m.interval = 100 * time.Millisecond
	}
	m.initPoller()
	if m.poller == nil {
		return
	}
	defer m.poller.close()

	m.poll()
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.poll()
		}
	}
}

func (m *gpuMonitor) initPoller() {
	m.initOnce.Do(func() {
		if poller, ok := newNVMLPoller(); ok {
			m.poller = poller
			return
		}
		if poller, ok := newSMIPoller(); ok {
			m.poller = poller
		}
	})
}

func (m *gpuMonitor) poll() {
	if m.poller == nil {
		return
	}
	sample, ok := m.poller.poll()
	if !ok {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.available = true
	m.source = m.poller.source()
	if sample.peakVRAMBytes > m.peakVRAM {
		m.peakVRAM = sample.peakVRAMBytes
	}
	m.sumMeanUtil += sample.meanUtil
	if sample.meanUtil > m.peakMeanUtil {
		m.peakMeanUtil = sample.meanUtil
	}
	if sample.processVRAMBytes != nil {
		m.hasProcessVRAM = true
		if *sample.processVRAMBytes > m.processVRAM {
			m.processVRAM = *sample.processVRAMBytes
		}
	}
	if sample.totalVisibleVRAMBytes != nil {
		m.hasTotalVisibleVRAM = true
		if *sample.totalVisibleVRAMBytes > m.totalVisibleVRAM {
			m.totalVisibleVRAM = *sample.totalVisibleVRAMBytes
		}
	}
	if len(sample.perGPUUsedBytes) > 0 {
		if m.perGPUUsed == nil {
			m.perGPUUsed = make(map[string]int64, len(sample.perGPUUsedBytes))
		}
		for key, value := range sample.perGPUUsedBytes {
			if value > m.perGPUUsed[key] {
				m.perGPUUsed[key] = value
			}
		}
	}
	if len(sample.perGPUFreeBytes) > 0 {
		if m.perGPUFree == nil {
			m.perGPUFree = make(map[string]int64, len(sample.perGPUFreeBytes))
		}
		for key, value := range sample.perGPUFreeBytes {
			m.perGPUFree[key] = value
		}
	}
	if sample.visibleGPUCount > m.visibleGPUCount {
		m.visibleGPUCount = sample.visibleGPUCount
	}
	m.samples++
}

func (m *gpuMonitor) stats() gpuStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.available || m.samples == 0 {
		return gpuStats{Available: false, Source: "unavailable"}
	}

	stats := gpuStats{
		Available:       true,
		Source:          firstNonEmpty(m.source, "unavailable"),
		PeakVRAMBytes:   int64Ptr(m.peakVRAM),
		CurrentVRAMBytes: int64Ptr(m.peakVRAM),
		AvgGPUUtil:      float64Ptr(m.sumMeanUtil / float64(m.samples)),
		PeakGPUUtil:     float64Ptr(m.peakMeanUtil),
		VisibleGPUCount: m.visibleGPUCount,
		SampleCount:     m.samples,
	}
	if m.hasProcessVRAM {
		stats.ProcessVRAMBytes = int64Ptr(m.processVRAM)
	}
	if m.hasTotalVisibleVRAM {
		stats.TotalVisibleVRAMBytes = int64Ptr(m.totalVisibleVRAM)
	}
	if len(m.perGPUUsed) > 0 {
		stats.PerGPUUsedBytes = maps.Clone(m.perGPUUsed)
	}
	if len(m.perGPUFree) > 0 {
		stats.PerGPUFreeBytes = maps.Clone(m.perGPUFree)
	}
	return stats
}

type smiPoller struct{}

func newSMIPoller() (gpuPoller, bool) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits")
	if err := cmd.Run(); err != nil {
		return nil, false
	}
	return &smiPoller{}, true
}

func (p *smiPoller) poll() (gpuSample, bool) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return gpuSample{}, false
	}
	peakVRAM, meanUtil, totalVisible, perGPUUsed, perGPUFree, visibleCount, ok := parseNvidiaSMI(string(out))
	if !ok {
		return gpuSample{}, false
	}

	sample := gpuSample{
		peakVRAMBytes:         peakVRAM,
		meanUtil:              meanUtil,
		totalVisibleVRAMBytes: totalVisible,
		perGPUUsedBytes:       perGPUUsed,
		perGPUFreeBytes:       perGPUFree,
		visibleGPUCount:       visibleCount,
	}
	if processVRAM, ok := queryProcessVRAMFromSMI(); ok {
		sample.processVRAMBytes = int64Ptr(processVRAM)
	}
	return sample, true
}

func (p *smiPoller) close() {}

func (p *smiPoller) source() string { return "nvidia-smi" }

func parseNvidiaSMI(out string) (int64, float64, *int64, map[string]int64, map[string]int64, int, bool) {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		return 0, 0, nil, nil, nil, 0, false
	}

	var totalMemMiB int64
	var totalVisibleMiB int64
	var totalUtil float64
	var count int
	perGPUUsed := make(map[string]int64)
	perGPUFree := make(map[string]int64)

	for _, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) < 5 {
			return 0, 0, nil, nil, nil, 0, false
		}
		index := strings.TrimSpace(parts[0])
		memUsed, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			return 0, 0, nil, nil, nil, 0, false
		}
		memFree, err := strconv.ParseInt(strings.TrimSpace(parts[2]), 10, 64)
		if err != nil {
			return 0, 0, nil, nil, nil, 0, false
		}
		memTotal, err := strconv.ParseInt(strings.TrimSpace(parts[3]), 10, 64)
		if err != nil {
			return 0, 0, nil, nil, nil, 0, false
		}
		util, err := strconv.ParseFloat(strings.TrimSpace(parts[4]), 64)
		if err != nil {
			return 0, 0, nil, nil, nil, 0, false
		}
		totalMemMiB += memUsed
		totalVisibleMiB += memTotal
		totalUtil += util
		perGPUUsed[index] = memUsed * 1024 * 1024
		perGPUFree[index] = memFree * 1024 * 1024
		count++
	}
	if count == 0 {
		return 0, 0, nil, nil, nil, 0, false
	}
	totalVisible := totalVisibleMiB * 1024 * 1024
	return totalMemMiB * 1024 * 1024, totalUtil / float64(count), int64Ptr(totalVisible), perGPUUsed, perGPUFree, count, true
}

func queryProcessVRAMFromSMI() (int64, bool) {
	pgrep := exec.Command("pgrep", "-f", "ollama")
	pidOut, err := pgrep.Output()
	if err != nil {
		return 0, false
	}

	pids := make(map[string]struct{})
	for _, pid := range strings.Fields(string(pidOut)) {
		pids[strings.TrimSpace(pid)] = struct{}{}
	}
	if len(pids) == 0 {
		return 0, false
	}

	cmd := exec.Command("nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return 0, false
	}

	var totalMiB int64
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			continue
		}
		pid := strings.TrimSpace(parts[0])
		if _, ok := pids[pid]; !ok {
			continue
		}
		used, err := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
		if err != nil {
			continue
		}
		totalMiB += used
	}
	if totalMiB <= 0 {
		return 0, false
	}
	return totalMiB * 1024 * 1024, true
}
