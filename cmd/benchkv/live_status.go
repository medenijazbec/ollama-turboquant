package main

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/term"
)

type liveConsoleRenderer struct {
	out       io.Writer
	enabled   bool
	isTTY     bool
	interval  time.Duration
	manager   *liveManager
	suite     string
	total     int
	stopCh    chan struct{}
	lastLines int
	mu        sync.Mutex
}

func newLiveConsoleRenderer(mode progressMode, interval time.Duration, suite string, total int, manager *liveManager) *liveConsoleRenderer {
	isTTY := term.IsTerminal(int(os.Stderr.Fd()))
	enabled := shouldEnableProgress(mode, isTTY)
	r := &liveConsoleRenderer{
		out:      os.Stderr,
		enabled:  enabled,
		isTTY:    isTTY,
		interval: interval,
		manager:  manager,
		suite:    suite,
		total:    total,
		stopCh:   make(chan struct{}),
	}
	if enabled {
		if interval <= 0 {
			interval = time.Second
		}
		go r.run()
	}
	return r
}

func (r *liveConsoleRenderer) run() {
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			r.render()
		case <-r.stopCh:
			r.render()
			return
		}
	}
}

func (r *liveConsoleRenderer) close() {
	if !r.enabled {
		return
	}
	close(r.stopCh)
}

func (r *liveConsoleRenderer) render() {
	if !r.enabled {
		return
	}
	samples := r.manager.snapshot()
	if len(samples) == 0 {
		return
	}
	lines := make([]string, 0, len(samples)+1)
	lines = append(lines, fmt.Sprintf("[suite %s] active=%d total=%d", r.suite, len(samples), r.total))
	for _, sample := range samples {
		lines = append(lines, renderLiveSampleLine(sample))
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.isTTY {
		if r.lastLines > 0 {
			fmt.Fprintf(r.out, "\x1b[%dA", r.lastLines)
		}
		for i, line := range lines {
			if i < len(lines)-1 {
				fmt.Fprintf(r.out, "\r%s\x1b[K\n", line)
			} else {
				fmt.Fprintf(r.out, "\r%s\x1b[K", line)
			}
		}
		r.lastLines = len(lines)
		return
	}
	for _, line := range lines[1:] {
		fmt.Fprintln(r.out, line)
	}
}

func renderLiveSampleLine(s liveSample) string {
	eta := "--:--:--"
	if s.ETASec != nil {
		eta = formatHMS(time.Duration(*s.ETASec * float64(time.Second)))
		if s.ETAConfidence != "" {
			eta += "(" + s.ETAConfidence + ")"
		}
	}
	progress := "progress_unknown"
	if s.ProgressKnown {
		progress = fmt.Sprintf("%.0f%%", s.ProgressPercent)
	}
	validate := firstNonEmpty(s.ValidationState, "pending")
	fallback := "no"
	if s.FallbackApplied {
		fallback = "yes"
	}
	gpuBits := renderLiveGPUCompact(s)
	return fmt.Sprintf("[%s %02d/%02d] host=%s model=%s workload=%s stage=%s ctx=%d/%d req=%s eff=%s fa=%t/%t elapsed=%s eta=%s p=%s rpt=%d/%d ladder=%d/%d p_tok/s=%.1f d_tok/s=%.1f gen=%d/%d ram=%s peak_ram=%s %s fallback=%s validate=%s",
		s.Suite,
		s.TestIndex,
		max(1, s.TestTotal),
		s.HostLabel,
		s.Model,
		s.Workload,
		s.Stage,
		s.RequestedContext,
		max(s.EffectiveContext, s.RequestedContext),
		s.RequestedKVMode,
		firstNonEmpty(s.EffectiveKVMode, s.RequestedKVMode),
		s.FlashAttentionRequested,
		s.FlashAttentionEffective,
		formatHMS(time.Duration(s.ElapsedSec*float64(time.Second))),
		eta,
		progress,
		max(1, s.Epoch),
		max(1, s.EpochTotal),
		max(1, s.LadderPosition),
		max(1, s.LadderTotal),
		s.CumulativePromptTPS,
		s.CumulativeDecodeTPS,
		s.GeneratedTokens,
		max(0, s.PromptTokensProcessed+s.GeneratedTokens),
		formatBytesMaybe(s.HostRAMUsedBytes),
		formatBytesMaybe(s.HostRAMPeakBytes),
		gpuBits,
		fallback,
		validate,
	)
}

func renderLiveGPUCompact(s liveSample) string {
	if len(s.GPUVRAMUsedBytesByGPU) == 0 {
		return "gpu=unavailable"
	}
	keys := make([]string, 0, len(s.GPUVRAMUsedBytesByGPU))
	for k := range s.GPUVRAMUsedBytesByGPU {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		used := s.GPUVRAMUsedBytesByGPU[k]
		free := s.GPUVRAMFreeBytesByGPU[k]
		util := s.GPUUtilPercentByGPU[k]
		parts = append(parts, fmt.Sprintf("gpu%s=%s/%s util=%.0f%%", k, formatBytes(used), formatBytes(used+free), util))
	}
	return strings.Join(parts, " ")
}

func formatBytesMaybe(v *int64) string {
	if v == nil {
		return "-"
	}
	return formatBytes(*v)
}

func formatBytes(v int64) string {
	const unit = int64(1024)
	if v < unit {
		return fmt.Sprintf("%dB", v)
	}
	div, exp := unit, 0
	for n := v / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%c", float64(v)/float64(div), "KMGTPE"[exp])
}
