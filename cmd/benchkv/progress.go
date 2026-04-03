package main

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"golang.org/x/term"
)

const progressHeartbeatInterval = 2 * time.Second

func newProgressTracker(mode progressMode, width, total int, debug bool) *progressTracker {
	enabled := shouldEnableProgress(mode, term.IsTerminal(int(os.Stderr.Fd())))
	if width <= 0 {
		width = 30
	}
	if total <= 0 {
		total = 1
	}
	tracker := &progressTracker{
		out:       os.Stderr,
		enabled:   enabled,
		debug:     debug,
		width:     width,
		total:     total,
		startedAt: time.Now(),
	}
	if tracker.enabled {
		tracker.ticker = time.NewTicker(progressHeartbeatInterval)
		tracker.stopCh = make(chan struct{})
		go tracker.runHeartbeat()
	}
	return tracker
}

func shouldEnableProgress(mode progressMode, isTTY bool) bool {
	switch mode {
	case progressOn:
		return true
	case progressOff:
		return false
	default:
		return isTTY
	}
}

func estimateTotalUnits(cfg config) int {
	total := len(cfg.Hosts)
	standardCells := buildStandardCells(cfg)
	total += len(standardCells) * (cfg.Warmup + cfg.Epochs)
	if cfg.Profile == "large-context" {
		total += len(standardCells) * (len(cfg.ContextLadder) - 1) * (cfg.Warmup + cfg.Epochs)
	}
	if slicesContainsWorkload(cfg.Workloads, workloadNearOOMStaircase) {
		staircaseCells := 0
		hostCount := len(cfg.Hosts)
		if cfg.Profile == "capacity" || cfg.Profile == "spill" {
			hostCount = 0
			for _, host := range cfg.Hosts {
				if host.KVSupportMode == hostKVSupportRequest {
					hostCount++
				}
			}
		}
		for _, numCtx := range cfg.NumCtx {
			for _, conc := range cfg.Concurrency {
				if _, ok := makeWorkloadSpec(cfg, workloadNearOOMStaircase, numCtx, conc); ok {
					staircaseCells++
				}
			}
		}
		total += hostCount * len(cfg.KVModes) * staircaseCells * (cfg.Warmup + cfg.Epochs)
	}
	if total <= 0 {
		return 1
	}
	return total
}

func estimateTotalUnitsWithPreflight(cfg config, preflights map[string]hostPreflight) int {
	total := len(cfg.Hosts)
	unitsPerCell := cfg.Warmup + cfg.Epochs

	for _, cell := range buildStandardCells(cfg) {
		if !supportedByPreflight(preflights, cell.Host.BaseURL, cell.KVMode) {
			continue
		}
		multiplier := 1
		if cfg.Profile == "large-context" {
			multiplier = max(len(cfg.ContextLadder), 1)
		}
		total += unitsPerCell * multiplier
	}

	if slicesContainsWorkload(cfg.Workloads, workloadNearOOMStaircase) {
		staircaseCells := 0
		for _, numCtx := range cfg.NumCtx {
			for _, conc := range cfg.Concurrency {
				if _, ok := makeWorkloadSpec(cfg, workloadNearOOMStaircase, numCtx, conc); ok {
					staircaseCells++
				}
			}
		}
		for _, host := range cfg.Hosts {
			if (cfg.Profile == "capacity" || cfg.Profile == "spill") && host.KVSupportMode != hostKVSupportRequest {
				continue
			}
			for _, kvMode := range cfg.KVModes {
				if !supportedByPreflight(preflights, host.BaseURL, kvMode) {
					continue
				}
				total += staircaseCells * unitsPerCell
			}
		}
	}

	if total <= 0 {
		return 1
	}
	return total
}

func supportedByPreflight(preflights map[string]hostPreflight, hostURL, kvMode string) bool {
	preflight, ok := preflights[hostURL]
	if !ok {
		return true
	}
	result, ok := preflight.KVSupport[kvMode]
	if !ok {
		return true
	}
	return result.Supported
}

func (p *progressTracker) SetCurrent(step progressStep) {
	p.mu.Lock()
	p.current = step
	p.renderLocked()
	p.mu.Unlock()
}

func (p *progressTracker) SetTotal(total int) {
	if total <= 0 {
		total = 1
	}
	p.mu.Lock()
	p.total = total
	if p.completed > p.total {
		p.completed = p.total
	}
	p.renderLocked()
	p.mu.Unlock()
}

func (p *progressTracker) Advance(n int) {
	if n <= 0 {
		return
	}
	p.mu.Lock()
	p.completed += n
	if p.completed > p.total {
		p.completed = p.total
	}
	p.renderLocked()
	p.mu.Unlock()
}

func (p *progressTracker) Skip(n int, msg string) {
	p.Status(msg)
	p.Advance(n)
}

func (p *progressTracker) Status(msg string) {
	if p.enabled {
		p.mu.Lock()
		p.clearLineLocked()
		fmt.Fprintln(p.out, msg)
		p.renderLocked()
		p.mu.Unlock()
		return
	}
	if p.debug {
		fmt.Fprintln(p.out, msg)
	}
}

func (p *progressTracker) Finish() {
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	if p.stopCh != nil {
		close(p.stopCh)
		p.stopCh = nil
	}
	if p.ticker != nil {
		p.ticker.Stop()
		p.ticker = nil
	}
	p.completed = p.total
	p.renderLocked()
	if p.enabled {
		fmt.Fprintln(p.out)
	}
	p.closed = true
	p.mu.Unlock()
}

func (p *progressTracker) render() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.renderLocked()
}

func (p *progressTracker) renderLocked() {
	if !p.enabled {
		return
	}
	line := p.line()
	padding := ""
	if extra := p.lastWidth - len(line); extra > 0 {
		padding = strings.Repeat(" ", extra)
	}
	fmt.Fprintf(p.out, "\r%s%s", line, padding)
	p.lastWidth = len(line)
}

func (p *progressTracker) clearLine() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.clearLineLocked()
}

func (p *progressTracker) clearLineLocked() {
	if !p.enabled {
		return
	}
	fmt.Fprintf(p.out, "\r%s\r", strings.Repeat(" ", p.lastWidth))
}

func (p *progressTracker) line() string {
	ratio := float64(p.completed) / float64(max(p.total, 1))
	filled := int(ratio * float64(p.width))
	if filled > p.width {
		filled = p.width
	}
	bar := strings.Repeat("#", filled) + strings.Repeat("-", p.width-filled)
	pct := int(ratio * 100)
	elapsed := time.Since(p.startedAt)
	eta := "--:--:--"
	if p.completed >= 2 && elapsed > 0 {
		rate := float64(p.completed) / elapsed.Seconds()
		if rate > 0 {
			remaining := float64(p.total-p.completed) / rate
			eta = formatHMS(time.Duration(remaining * float64(time.Second)))
		}
	}
	pulse := [...]string{"|", "/", "-", "\\"}[p.pulseIndex%4]
	return fmt.Sprintf(
		"[%s] %d%% | tick %s | %s | %s | %s | ctx=%d | conc=%d | %s %d/%d | elapsed %s | eta %s",
		bar,
		pct,
		pulse,
		firstNonEmpty(p.current.HostLabel, "-"),
		firstNonEmpty(p.current.KVMode, "-"),
		firstNonEmpty(p.current.Workload, p.current.Phase, "-"),
		p.current.NumCtx,
		p.current.Concurrency,
		epochLabel(p.current.Warmup),
		p.current.Epoch,
		max(p.current.EpochTotal, 1),
		formatHMS(elapsed),
		eta,
	)
}

func (p *progressTracker) runHeartbeat() {
	for {
		select {
		case <-p.stopCh:
			return
		case <-p.ticker.C:
			p.mu.Lock()
			if p.closed {
				p.mu.Unlock()
				return
			}
			p.pulseIndex++
			p.renderLocked()
			p.mu.Unlock()
		}
	}
}

func formatHMS(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	total := int(d.Seconds())
	h := total / 3600
	m := (total % 3600) / 60
	s := total % 60
	return fmt.Sprintf("%02d:%02d:%02d", h, m, s)
}

func epochLabel(warmup bool) string {
	if warmup {
		return "warmup"
	}
	return "epoch"
}

func slicesContainsWorkload(workloads []workloadName, target workloadName) bool {
	for _, workload := range workloads {
		if workload == target {
			return true
		}
	}
	return false
}

func newTestProgressTracker(out io.Writer, enabled bool, width, total int, debug bool) *progressTracker {
	if width <= 0 {
		width = 30
	}
	if total <= 0 {
		total = 1
	}
	return &progressTracker{
		out:       out,
		enabled:   enabled,
		debug:     debug,
		width:     width,
		total:     total,
		startedAt: time.Now().Add(-5 * time.Second),
	}
}
