package main

import (
	"context"
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
	peakVRAMBytes    int64
	meanUtil         float64
	processVRAMBytes *int64
}

type gpuMonitor struct {
	interval time.Duration
	enabled  bool

	initOnce sync.Once
	poller   gpuPoller

	mu             sync.Mutex
	available      bool
	source         string
	peakVRAM       int64
	sumMeanUtil    float64
	peakMeanUtil   float64
	processVRAM    int64
	hasProcessVRAM bool
	samples        int
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
	m.samples++
}

func (m *gpuMonitor) stats() gpuStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.available || m.samples == 0 {
		return gpuStats{Available: false, Source: "unavailable"}
	}

	stats := gpuStats{
		Available:     true,
		Source:        firstNonEmpty(m.source, "unavailable"),
		PeakVRAMBytes: int64Ptr(m.peakVRAM),
		AvgGPUUtil:    float64Ptr(m.sumMeanUtil / float64(m.samples)),
		PeakGPUUtil:   float64Ptr(m.peakMeanUtil),
		SampleCount:   m.samples,
	}
	if m.hasProcessVRAM {
		stats.ProcessVRAMBytes = int64Ptr(m.processVRAM)
	}
	return stats
}

type smiPoller struct{}

func newSMIPoller() (gpuPoller, bool) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits")
	if err := cmd.Run(); err != nil {
		return nil, false
	}
	return &smiPoller{}, true
}

func (p *smiPoller) poll() (gpuSample, bool) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil {
		return gpuSample{}, false
	}
	peakVRAM, meanUtil, ok := parseNvidiaSMI(string(out))
	if !ok {
		return gpuSample{}, false
	}

	sample := gpuSample{
		peakVRAMBytes: peakVRAM,
		meanUtil:      meanUtil,
	}
	if processVRAM, ok := queryProcessVRAMFromSMI(); ok {
		sample.processVRAMBytes = int64Ptr(processVRAM)
	}
	return sample, true
}

func (p *smiPoller) close() {}

func (p *smiPoller) source() string { return "nvidia-smi" }

func parseNvidiaSMI(out string) (int64, float64, bool) {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		return 0, 0, false
	}

	var totalMemMiB int64
	var totalUtil float64
	var count int

	for _, line := range lines {
		parts := strings.Split(line, ",")
		if len(parts) < 2 {
			return 0, 0, false
		}
		memUsed, err := strconv.ParseInt(strings.TrimSpace(parts[0]), 10, 64)
		if err != nil {
			return 0, 0, false
		}
		util, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err != nil {
			return 0, 0, false
		}
		totalMemMiB += memUsed
		totalUtil += util
		count++
	}
	if count == 0 {
		return 0, 0, false
	}

	return totalMemMiB * 1024 * 1024, totalUtil / float64(count), true
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
