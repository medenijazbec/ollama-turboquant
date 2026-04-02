package main

import (
	"bufio"
	"context"
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

	mu         sync.Mutex
	available  bool
	source     string
	currentRAM int64
	peakRAM    int64
	processRSS int64
	samples    int
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
	used, source, ok := readHostRAMUsed()
	if !ok {
		return
	}
	rss := captureOllamaProcessRSS()

	m.mu.Lock()
	defer m.mu.Unlock()
	m.available = true
	m.source = source
	m.currentRAM = used
	if used > m.peakRAM {
		m.peakRAM = used
	}
	if rss > m.processRSS {
		m.processRSS = rss
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
		Available:        true,
		Source:           firstNonEmpty(m.source, "unavailable"),
		HostRAMUsedBytes: int64Ptr(m.currentRAM),
		PeakHostRAMBytes: int64Ptr(m.peakRAM),
		SampleCount:      m.samples,
	}
	if m.processRSS > 0 {
		stats.ProcessRSSBytes = int64Ptr(m.processRSS)
	}
	return stats
}

func readHostRAMUsed() (int64, string, bool) {
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
		return 0, "", false
	}
	return used, "proc-meminfo", true
}

func readHostRAMUsedFromFree() (int64, string, bool) {
	cmd := exec.Command("free", "-b")
	out, err := cmd.Output()
	if err != nil {
		return 0, "", false
	}

	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 7 || fields[0] != "Mem:" {
			continue
		}
		total, err1 := strconv.ParseInt(fields[1], 10, 64)
		available, err2 := strconv.ParseInt(fields[6], 10, 64)
		if err1 != nil || err2 != nil {
			return 0, "", false
		}
		return total - available, "free", true
	}
	return 0, "", false
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
