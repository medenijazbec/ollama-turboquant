package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

type hostMetricsMonitor struct {
	interval time.Duration
	enabled  bool

	mu           sync.Mutex
	available    bool
	source       string
	baselineRAM  int64
	currentRAM   int64
	peakRAM      int64
	processRSS   int64
	selfRSS      int64
	availableRAM int64
	cpuUtil      float64
	hasCPUUtil   bool
	lastCPUIdle  uint64
	lastCPUTotal uint64
	samples      int
}

func newHostMetricsMonitor(interval time.Duration, enabled bool) *hostMetricsMonitor {
	return &hostMetricsMonitor{
		interval: interval,
		enabled:  enabled,
	}
}

func (m *hostMetricsMonitor) run(ctx context.Context) {
	if !m.enabled {
		return
	}
	if m.interval <= 0 {
		m.interval = 100 * time.Millisecond
	}

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

func (m *hostMetricsMonitor) poll() {
	used, available, source, ok := readHostRAMUsed()
	if !ok {
		return
	}
	rss := captureOllamaProcessRSS()
	selfRSS := captureSelfRSS()
	cpuUtil, cpuOK := readCPUUtilization(&m.lastCPUIdle, &m.lastCPUTotal)

	m.mu.Lock()
	defer m.mu.Unlock()
	m.available = true
	m.source = source
	if m.baselineRAM == 0 {
		m.baselineRAM = used
	}
	m.currentRAM = used
	m.availableRAM = available
	if used > m.peakRAM {
		m.peakRAM = used
	}
	if rss > m.processRSS {
		m.processRSS = rss
	}
	if selfRSS > 0 {
		m.selfRSS = selfRSS
	}
	if cpuOK {
		m.cpuUtil = cpuUtil
		m.hasCPUUtil = true
	}
	m.samples++
}

func (m *hostMetricsMonitor) stats() hostMemoryStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.available || m.samples == 0 {
		return hostMemoryStats{Available: false, Source: "unavailable"}
	}

	stats := hostMemoryStats{
		Available: true,
		Source:    firstNonEmpty(m.source, "unavailable"),
		HostRAMBeforeBytes: func() *int64 {
			if m.baselineRAM > 0 {
				return int64Ptr(m.baselineRAM)
			}
			return nil
		}(),
		HostRAMUsedBytes: int64Ptr(m.currentRAM),
		PeakHostRAMBytes: int64Ptr(m.peakRAM),
		SampleCount:      m.samples,
	}
	if m.baselineRAM > 0 && m.peakRAM >= m.baselineRAM {
		stats.PeakHostRAMDeltaBytes = int64Ptr(m.peakRAM - m.baselineRAM)
	}
	if m.processRSS > 0 {
		stats.ProcessRSSBytes = int64Ptr(m.processRSS)
	}
	if m.selfRSS > 0 {
		stats.SelfRSSBytes = int64Ptr(m.selfRSS)
	}
	if m.availableRAM > 0 {
		stats.SystemAvailableBytes = int64Ptr(m.availableRAM)
	}
	if m.hasCPUUtil {
		stats.CPUUtilPercent = float64Ptr(m.cpuUtil)
	}
	return stats
}

func readHostRAMUsed() (int64, int64, string, bool) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return readHostRAMUsedFromFree()
	}
	defer file.Close()

	var memTotalKB int64
	var memAvailableKB int64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			memTotalKB, _ = strconv.ParseInt(fields[1], 10, 64)
		case "MemAvailable:":
			memAvailableKB, _ = strconv.ParseInt(fields[1], 10, 64)
		}
	}
	if memTotalKB <= 0 || memAvailableKB <= 0 {
		return readHostRAMUsedFromFree()
	}

	used := (memTotalKB - memAvailableKB) * 1024
	if used < 0 {
		return 0, 0, "", false
	}
	return used, memAvailableKB * 1024, "proc-meminfo", true
}

func readHostRAMUsedFromFree() (int64, int64, string, bool) {
	cmd := exec.Command("free", "-b")
	out, err := cmd.Output()
	if err != nil {
		return 0, 0, "", false
	}

	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 7 || fields[0] != "Mem:" {
			continue
		}
		total, err1 := strconv.ParseInt(fields[1], 10, 64)
		available, err2 := strconv.ParseInt(fields[6], 10, 64)
		if err1 != nil || err2 != nil {
			return 0, 0, "", false
		}
		return total - available, available, "free", true
	}
	return 0, 0, "", false
}

func captureOllamaProcessRSS() int64 {
	pgrep := exec.Command("pgrep", "-f", "ollama")
	out, err := pgrep.Output()
	if err != nil {
		return -1
	}
	pids := strings.Fields(string(out))
	var totalRSS int64
	for _, pid := range pids {
		cmd := exec.Command("ps", "-o", "rss=", "-p", pid)
		raw, err := cmd.Output()
		if err != nil {
			continue
		}
		rssKB, err := strconv.ParseInt(strings.TrimSpace(string(raw)), 10, 64)
		if err != nil {
			continue
		}
		totalRSS += rssKB * 1024
	}
	if totalRSS <= 0 {
		return -1
	}
	return totalRSS
}

func captureSelfRSS() int64 {
	data, err := os.ReadFile("/proc/self/status")
	if err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.HasPrefix(line, "VmRSS:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					if rssKB, parseErr := strconv.ParseInt(fields[1], 10, 64); parseErr == nil {
						return rssKB * 1024
					}
				}
			}
		}
	}
	pid := os.Getpid()
	cmd := exec.Command("ps", "-o", "rss=", "-p", fmt.Sprintf("%d", pid))
	out, err := cmd.Output()
	if err != nil {
		return -1
	}
	rssKB, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return -1
	}
	return rssKB * 1024
}

func readCPUUtilization(lastIdle, lastTotal *uint64) (float64, bool) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, false
	}
	lines := strings.Split(string(data), "\n")
	if len(lines) == 0 {
		return 0, false
	}
	fields := strings.Fields(lines[0])
	if len(fields) < 8 || fields[0] != "cpu" {
		return 0, false
	}
	var vals []uint64
	for _, field := range fields[1:] {
		v, parseErr := strconv.ParseUint(field, 10, 64)
		if parseErr != nil {
			return 0, false
		}
		vals = append(vals, v)
	}
	var total uint64
	for _, v := range vals {
		total += v
	}
	idle := vals[3]
	if *lastTotal == 0 {
		*lastIdle = idle
		*lastTotal = total
		return 0, false
	}
	totalDelta := total - *lastTotal
	idleDelta := idle - *lastIdle
	*lastIdle = idle
	*lastTotal = total
	if totalDelta == 0 {
		return 0, false
	}
	return (1 - float64(idleDelta)/float64(totalDelta)) * 100, true
}
