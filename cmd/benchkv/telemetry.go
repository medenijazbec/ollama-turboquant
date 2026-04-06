package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type liveSample struct {
	Suite                     string             `json:"suite"`
	TestID                    string             `json:"test_id"`
	TestIndex                 int                `json:"test_index"`
	TestTotal                 int                `json:"test_total"`
	Host                      string             `json:"host"`
	HostLabel                 string             `json:"host_label"`
	Model                     string             `json:"model"`
	Workload                  string             `json:"workload"`
	RequestedKVMode           string             `json:"requested_kv_mode"`
	EffectiveKVMode           string             `json:"effective_kv_mode,omitempty"`
	RequestedCacheTypeK       string             `json:"requested_cache_type_k,omitempty"`
	RequestedCacheTypeV       string             `json:"requested_cache_type_v,omitempty"`
	EffectiveCacheTypeK       string             `json:"effective_cache_type_k,omitempty"`
	EffectiveCacheTypeV       string             `json:"effective_cache_type_v,omitempty"`
	RequestedContext          int                `json:"requested_context"`
	EffectiveContext          int                `json:"effective_context,omitempty"`
	FlashAttentionRequested   bool               `json:"flash_attention_requested"`
	FlashAttentionEffective   bool               `json:"flash_attention_effective"`
	LadderPosition            int                `json:"ladder_position,omitempty"`
	LadderTotal               int                `json:"ladder_total,omitempty"`
	Epoch                     int                `json:"epoch"`
	EpochTotal                int                `json:"epoch_total"`
	Repeat                    int                `json:"repeat"`
	RepeatTotal               int                `json:"repeat_total"`
	Timestamp                 time.Time          `json:"timestamp"`
	Stage                     string             `json:"stage"`
	RequestStatus             string             `json:"request_status"`
	ElapsedSec                float64            `json:"elapsed_sec"`
	StageElapsedSec           float64            `json:"stage_elapsed_sec"`
	ProgressPercent           float64            `json:"progress_percent,omitempty"`
	ProgressKnown             bool               `json:"progress_known"`
	ETASec                    *float64           `json:"eta_sec,omitempty"`
	ETAConfidence             string             `json:"eta_confidence,omitempty"`
	TimeoutBudgetRemainingSec *float64           `json:"timeout_budget_remaining_sec,omitempty"`
	RollingPromptTPS          float64            `json:"rolling_prompt_tps,omitempty"`
	RollingDecodeTPS          float64            `json:"rolling_decode_tps,omitempty"`
	CumulativePromptTPS       float64            `json:"cumulative_prompt_tps,omitempty"`
	CumulativeDecodeTPS       float64            `json:"cumulative_decode_tps,omitempty"`
	GeneratedTokens           int                `json:"generated_tokens"`
	PromptTokensProcessed     int                `json:"prompt_tokens_processed"`
	TotalTokensProcessed      int                `json:"total_tokens_processed"`
	HostRAMUsedBytes          *int64             `json:"host_ram_used_bytes,omitempty"`
	HostRAMPeakBytes          *int64             `json:"host_ram_peak_bytes,omitempty"`
	BenchmarkProcessRSSBytes  *int64             `json:"benchmark_process_rss_bytes,omitempty"`
	OllamaProcessRSSBytes     *int64             `json:"ollama_process_rss_bytes,omitempty"`
	CPUUtilPercent            *float64           `json:"cpu_util_percent,omitempty"`
	GPUSource                 string             `json:"gpu_source,omitempty"`
	GPUCount                  int                `json:"gpu_count,omitempty"`
	GPUVRAMUsedBytesTotal     *int64             `json:"gpu_vram_used_bytes_total,omitempty"`
	GPUVRAMPeakBytesTotal     *int64             `json:"gpu_vram_peak_bytes_total,omitempty"`
	GPUVRAMUsedBytesByGPU     map[string]int64   `json:"gpu_vram_used_bytes_by_gpu,omitempty"`
	GPUVRAMFreeBytesByGPU     map[string]int64   `json:"gpu_vram_free_bytes_by_gpu,omitempty"`
	GPUVRAMPeakBytesByGPU     map[string]int64   `json:"gpu_vram_peak_bytes_by_gpu,omitempty"`
	GPUUtilPercentByGPU       map[string]float64 `json:"gpu_util_percent_by_gpu,omitempty"`
	GPUTempCByGPU             map[string]float64 `json:"gpu_temp_c_by_gpu,omitempty"`
	GPUPowerWByGPU            map[string]float64 `json:"gpu_power_w_by_gpu,omitempty"`
	CorruptionFlag            bool               `json:"corruption_flag"`
	CorruptionClass           string             `json:"corruption_class,omitempty"`
	FallbackApplied           bool               `json:"fallback_applied"`
	FallbackReason            string             `json:"fallback_reason,omitempty"`
	ValidationState           string             `json:"validation_state,omitempty"`
	ValidationKind            string             `json:"validation_kind,omitempty"`
	TruncationFlag            bool               `json:"truncation_flag"`
	RetryCount                int                `json:"retry_count,omitempty"`
	EndpointHealth            string             `json:"endpoint_health,omitempty"`
	Parallelism               int                `json:"parallelism,omitempty"`
	ActiveSlots               int                `json:"active_slots,omitempty"`
	CurrentContextDepth       int                `json:"current_context_depth,omitempty"`
	KVBytesEstimate           *int64             `json:"kv_bytes_estimate,omitempty"`
	TerminalState             string             `json:"terminal_state,omitempty"`
	Error                     string             `json:"error,omitempty"`
}

type liveSessionSummary struct {
	TelemetryPath         string
	PeakGPUVRAMBytesTotal *int64
	PeakGPUVRAMBytesByGPU string
	AvgPromptTPS          *float64
	AvgDecodeTPS          *float64
	MaxPromptTPS          *float64
	MaxDecodeTPS          *float64
	ETAConfidence         string
	StageDurations        string
	ProgressSamples       int
	LiveStatusEnabled     bool
	TelemetrySamplingSec  int
}

type liveWorkerState struct {
	generatedTokens int
	done            bool
}

type liveSession struct {
	mu                   sync.Mutex
	manager              *liveManager
	writer               *telemetryWriter
	id                   string
	testIndex            int
	testTotal            int
	suite                string
	host                 string
	hostLabel            string
	model                string
	workload             string
	requestedKVMode      string
	effectiveKVMode      string
	requestedCacheTypeK  string
	requestedCacheTypeV  string
	effectiveCacheTypeK  string
	effectiveCacheTypeV  string
	requestedContext     int
	effectiveContext     int
	faRequested          bool
	faEffective          bool
	ladderPosition       int
	ladderTotal          int
	epoch                int
	epochTotal           int
	repeat               int
	repeatTotal          int
	startedAt            time.Time
	stage                string
	stageStartedAt       time.Time
	validationState      string
	validationKind       string
	fallbackApplied      bool
	fallbackReason       string
	corruptionClass      string
	truncationFlag       bool
	endpointHealth       string
	errorText            string
	terminalState        string
	timeout              time.Duration
	promptTarget         int
	maxTokens            int
	promptTokens         int
	finalGeneratedTokens int
	promptSeen           bool
	workers              map[int]*liveWorkerState
	hostMonitor          *hostMetricsMonitor
	gpuMonitor           *gpuMonitor
	kvBytesEstimate      *int64
	progressSamples      int
	sumPromptTPS         float64
	sumDecodeTPS         float64
	maxPromptTPS         float64
	maxDecodeTPS         float64
	etaConfidence        string
	stageDurations       map[string]time.Duration
	lastSampleAt         time.Time
	lastSample           liveSample
	closed               bool
	displayEnabled       bool
	sampleEvery          time.Duration
	writeEvery           time.Duration
	doneCh               chan struct{}
}

func newLiveSession(manager *liveManager, opts liveSessionOptions) (*liveSession, error) {
	path := telemetryPathFor(manager.cfg.TelemetryDir, opts)
	var writer *telemetryWriter
	var err error
	if manager.cfg.LiveTelemetry {
		if err := ensureDir(filepath.Dir(path)); err != nil {
			return nil, err
		}
		writer, err = newTelemetryWriter(path)
		if err != nil {
			return nil, err
		}
	}
	s := &liveSession{
		manager:             manager,
		writer:              writer,
		id:                  opts.TestID,
		testIndex:           opts.TestIndex,
		testTotal:           opts.TestTotal,
		suite:               opts.Suite,
		host:                opts.Host,
		hostLabel:           opts.HostLabel,
		model:               opts.Model,
		workload:            opts.Workload,
		requestedKVMode:     opts.RequestedKVMode,
		effectiveKVMode:     opts.RequestedKVMode,
		requestedCacheTypeK: opts.RequestedCacheTypeK,
		requestedCacheTypeV: opts.RequestedCacheTypeV,
		effectiveCacheTypeK: opts.RequestedCacheTypeK,
		effectiveCacheTypeV: opts.RequestedCacheTypeV,
		requestedContext:    opts.RequestedContext,
		effectiveContext:    opts.RequestedContext,
		faRequested:         opts.FARequested,
		ladderPosition:      opts.LadderPosition,
		ladderTotal:         opts.LadderTotal,
		epoch:               opts.Epoch,
		epochTotal:          opts.EpochTotal,
		repeat:              opts.Repeat,
		repeatTotal:         opts.RepeatTotal,
		startedAt:           time.Now(),
		stage:               "queued",
		stageStartedAt:      time.Now(),
		endpointHealth:      "healthy",
		timeout:             opts.Timeout,
		promptTarget:        opts.PromptTarget,
		maxTokens:           opts.MaxTokens,
		workers:             map[int]*liveWorkerState{},
		stageDurations:      map[string]time.Duration{},
		displayEnabled:      opts.DisplayEnabled,
		sampleEvery:         manager.cfg.LiveSampleInterval,
		writeEvery:          manager.cfg.LiveWriteInterval,
		doneCh:              make(chan struct{}),
	}
	manager.register(s)
	go s.run()
	return s, nil
}

type liveSessionOptions struct {
	Suite               string
	TestID              string
	TestIndex           int
	TestTotal           int
	Host                string
	HostLabel           string
	Model               string
	Workload            string
	RequestedKVMode     string
	RequestedCacheTypeK string
	RequestedCacheTypeV string
	RequestedContext    int
	FARequested         bool
	LadderPosition      int
	LadderTotal         int
	Epoch               int
	EpochTotal          int
	Repeat              int
	RepeatTotal         int
	Timeout             time.Duration
	PromptTarget        int
	MaxTokens           int
	DisplayEnabled      bool
}

func (s *liveSession) run() {
	ticker := time.NewTicker(s.sampleEvery)
	defer ticker.Stop()
	s.sampleAndWrite("")
	for {
		select {
		case <-ticker.C:
			s.sampleAndWrite("")
		case <-s.doneCh:
			s.sampleAndWrite("")
			return
		}
	}
}

func (s *liveSession) sampleAndWrite(terminalState string) {
	s.mu.Lock()
	sample := s.buildSampleLocked(terminalState)
	s.lastSample = sample
	s.lastSampleAt = time.Now()
	s.progressSamples++
	if sample.CumulativePromptTPS > 0 {
		s.sumPromptTPS += sample.CumulativePromptTPS
		if sample.CumulativePromptTPS > s.maxPromptTPS {
			s.maxPromptTPS = sample.CumulativePromptTPS
		}
	}
	if sample.CumulativeDecodeTPS > 0 {
		s.sumDecodeTPS += sample.CumulativeDecodeTPS
		if sample.CumulativeDecodeTPS > s.maxDecodeTPS {
			s.maxDecodeTPS = sample.CumulativeDecodeTPS
		}
	}
	writer := s.writer
	s.mu.Unlock()
	if writer != nil {
		_ = writer.write(sample)
	}
}

func (s *liveSession) buildSampleLocked(terminalState string) liveSample {
	now := time.Now()
	stageElapsed := now.Sub(s.stageStartedAt)
	elapsed := now.Sub(s.startedAt)
	generated := 0
	for _, w := range s.workers {
		generated += w.generatedTokens
	}
	if s.finalGeneratedTokens > generated {
		generated = s.finalGeneratedTokens
	}
	promptTokens := s.promptTokens
	progressPct := 0.0
	progressKnown := false
	rollingPromptTPS := 0.0
	rollingDecodeTPS := 0.0
	cumulativePromptTPS := 0.0
	cumulativeDecodeTPS := 0.0
	if s.promptSeen && elapsed > 0 && promptTokens > 0 {
		cumulativePromptTPS = float64(promptTokens) / elapsed.Seconds()
		rollingPromptTPS = cumulativePromptTPS
	}
	if elapsed > 0 && generated > 0 {
		cumulativeDecodeTPS = float64(generated) / elapsed.Seconds()
		rollingDecodeTPS = cumulativeDecodeTPS
	}
	eta, etaConfidence, etaProgressKnown := estimateStageETA(s.stage, promptTokens, s.promptTarget, generated, s.maxTokens, rollingPromptTPS, rollingDecodeTPS, stageElapsed)
	if etaConfidence != "" {
		s.etaConfidence = etaConfidence
	}
	if s.stage == "decode" && s.maxTokens > 0 {
		progressPct = (float64(generated) / float64(s.maxTokens)) * 100
		progressKnown = true
	} else if s.stage == "prefill" && s.promptTarget > 0 && promptTokens > 0 {
		progressPct = (float64(promptTokens) / float64(s.promptTarget)) * 100
		progressKnown = true
	} else {
		progressKnown = etaProgressKnown
	}
	hostStats := hostMemoryStats{Source: "unavailable"}
	if s.hostMonitor != nil {
		hostStats = s.hostMonitor.stats()
	}
	gpuStats := gpuStats{Source: "unavailable"}
	if s.gpuMonitor != nil {
		gpuStats = s.gpuMonitor.stats()
	}
	var etaSec *float64
	if eta != nil {
		v := eta.Seconds()
		etaSec = &v
	}
	var timeoutRemaining *float64
	if s.timeout > 0 {
		v := s.timeout.Seconds() - elapsed.Seconds()
		if v < 0 {
			v = 0
		}
		timeoutRemaining = &v
	}
	return liveSample{
		Suite:                     s.suite,
		TestID:                    s.id,
		TestIndex:                 s.testIndex,
		TestTotal:                 s.testTotal,
		Host:                      s.host,
		HostLabel:                 s.hostLabel,
		Model:                     s.model,
		Workload:                  s.workload,
		RequestedKVMode:           s.requestedKVMode,
		EffectiveKVMode:           s.effectiveKVMode,
		RequestedCacheTypeK:       s.requestedCacheTypeK,
		RequestedCacheTypeV:       s.requestedCacheTypeV,
		EffectiveCacheTypeK:       s.effectiveCacheTypeK,
		EffectiveCacheTypeV:       s.effectiveCacheTypeV,
		RequestedContext:          s.requestedContext,
		EffectiveContext:          max(s.effectiveContext, s.requestedContext),
		FlashAttentionRequested:   s.faRequested,
		FlashAttentionEffective:   s.faEffective,
		LadderPosition:            s.ladderPosition,
		LadderTotal:               s.ladderTotal,
		Epoch:                     s.epoch,
		EpochTotal:                s.epochTotal,
		Repeat:                    s.repeat,
		RepeatTotal:               s.repeatTotal,
		Timestamp:                 now.UTC(),
		Stage:                     s.stage,
		RequestStatus:             s.stage,
		ElapsedSec:                elapsed.Seconds(),
		StageElapsedSec:           stageElapsed.Seconds(),
		ProgressPercent:           progressPct,
		ProgressKnown:             progressKnown,
		ETASec:                    etaSec,
		ETAConfidence:             firstNonEmpty(etaConfidence, s.etaConfidence),
		TimeoutBudgetRemainingSec: timeoutRemaining,
		RollingPromptTPS:          rollingPromptTPS,
		RollingDecodeTPS:          rollingDecodeTPS,
		CumulativePromptTPS:       cumulativePromptTPS,
		CumulativeDecodeTPS:       cumulativeDecodeTPS,
		GeneratedTokens:           generated,
		PromptTokensProcessed:     promptTokens,
		TotalTokensProcessed:      promptTokens + generated,
		HostRAMUsedBytes:          hostStats.HostRAMUsedBytes,
		HostRAMPeakBytes:          hostStats.PeakHostRAMBytes,
		BenchmarkProcessRSSBytes:  hostStats.SelfRSSBytes,
		OllamaProcessRSSBytes:     hostStats.ProcessRSSBytes,
		CPUUtilPercent:            hostStats.CPUUtilPercent,
		GPUSource:                 gpuStats.Source,
		GPUCount:                  gpuStats.VisibleGPUCount,
		GPUVRAMUsedBytesTotal:     gpuStats.CurrentVRAMBytes,
		GPUVRAMPeakBytesTotal:     gpuStats.PeakVRAMBytes,
		GPUVRAMUsedBytesByGPU:     cloneInt64Map(gpuStats.PerGPUUsedBytes),
		GPUVRAMFreeBytesByGPU:     cloneInt64Map(gpuStats.PerGPUFreeBytes),
		GPUVRAMPeakBytesByGPU:     cloneInt64Map(gpuStats.PeakPerGPUUsedBytes),
		GPUUtilPercentByGPU:       cloneFloat64Map(gpuStats.PerGPUUtilPercent),
		GPUTempCByGPU:             cloneFloat64Map(gpuStats.PerGPUTempC),
		GPUPowerWByGPU:            cloneFloat64Map(gpuStats.PerGPUPowerW),
		CorruptionFlag:            s.corruptionClass != "",
		CorruptionClass:           s.corruptionClass,
		FallbackApplied:           s.fallbackApplied,
		FallbackReason:            s.fallbackReason,
		ValidationState:           s.validationState,
		ValidationKind:            s.validationKind,
		TruncationFlag:            s.truncationFlag,
		EndpointHealth:            s.endpointHealth,
		Parallelism:               len(s.workers),
		ActiveSlots:               len(s.workers),
		CurrentContextDepth:       promptTokens + generated,
		KVBytesEstimate:           s.kvBytesEstimate,
		TerminalState:             firstNonEmpty(terminalState, s.terminalState),
		Error:                     s.errorText,
	}
}

func (s *liveSession) SetStage(stage string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || stage == "" || s.stage == stage {
		return
	}
	now := time.Now()
	s.stageDurations[s.stage] += now.Sub(s.stageStartedAt)
	s.stage = stage
	s.stageStartedAt = now
}

func (s *liveSession) UpdateWorker(workerIndex int, generatedDelta int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	w := s.workers[workerIndex]
	if w == nil {
		w = &liveWorkerState{}
		s.workers[workerIndex] = w
	}
	if generatedDelta > 0 {
		w.generatedTokens += generatedDelta
	}
}

func (s *liveSession) SetPromptSeen(promptTokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.promptSeen = true
	if promptTokens > s.promptTokens {
		s.promptTokens = promptTokens
	}
}

func (s *liveSession) SetEffectiveRuntime(effectiveMode, effectiveK, effectiveV string, faEffective bool, fallbackApplied bool, fallbackReason string, kvBytes *int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.effectiveKVMode = firstNonEmpty(effectiveMode, s.effectiveKVMode)
	s.effectiveCacheTypeK = firstNonEmpty(effectiveK, s.effectiveCacheTypeK)
	s.effectiveCacheTypeV = firstNonEmpty(effectiveV, s.effectiveCacheTypeV)
	s.faEffective = faEffective
	s.fallbackApplied = fallbackApplied
	s.fallbackReason = fallbackReason
	if kvBytes != nil {
		s.kvBytesEstimate = kvBytes
	}
}

func (s *liveSession) SetValidation(kind, state string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.validationKind = kind
	s.validationState = state
}

func (s *liveSession) SetValidationResult(corruptionClass string, truncation bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.corruptionClass = corruptionClass
	s.truncationFlag = truncation
}

func (s *liveSession) SetHostAndGPUMonitors(host *hostMetricsMonitor, gpu *gpuMonitor) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hostMonitor = host
	s.gpuMonitor = gpu
}

func (s *liveSession) Close(terminalState, errText string, finalPromptTokens, finalGenerated int) liveSessionSummary {
	s.mu.Lock()
	if s.closed {
		defer s.mu.Unlock()
		return s.summaryLocked()
	}
	s.closed = true
	s.errorText = errText
	s.terminalState = terminalState
	if finalPromptTokens > s.promptTokens {
		s.promptTokens = finalPromptTokens
	}
	if finalGenerated > s.finalGeneratedTokens {
		s.finalGeneratedTokens = finalGenerated
	}
	now := time.Now()
	s.stageDurations[s.stage] += now.Sub(s.stageStartedAt)
	close(s.doneCh)
	s.mu.Unlock()
	if s.writer != nil {
		_ = s.writer.close()
	}
	s.manager.unregister(s.id)
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.summaryLocked()
}

func (s *liveSession) summaryLocked() liveSessionSummary {
	var avgPrompt *float64
	var avgDecode *float64
	if s.progressSamples > 0 && s.sumPromptTPS > 0 {
		v := s.sumPromptTPS / float64(s.progressSamples)
		avgPrompt = &v
	}
	if s.progressSamples > 0 && s.sumDecodeTPS > 0 {
		v := s.sumDecodeTPS / float64(s.progressSamples)
		avgDecode = &v
	}
	var maxPrompt *float64
	var maxDecode *float64
	if s.maxPromptTPS > 0 {
		v := s.maxPromptTPS
		maxPrompt = &v
	}
	if s.maxDecodeTPS > 0 {
		v := s.maxDecodeTPS
		maxDecode = &v
	}
	var peakTotal *int64
	peakByGPU := ""
	if s.gpuMonitor != nil {
		stats := s.gpuMonitor.stats()
		peakTotal = stats.PeakVRAMBytes
		peakByGPU = formatInt64Map(stats.PeakPerGPUUsedBytes, "gpu")
	}
	telemetrySamplingSec := int(s.sampleEvery / time.Second)
	if telemetrySamplingSec < 1 {
		telemetrySamplingSec = 1
	}
	return liveSessionSummary{
		TelemetryPath: func() string {
			if s.writer != nil {
				return s.writer.path
			}
			return telemetryPathFor(s.manager.cfg.TelemetryDir, liveSessionOptions{Suite: s.suite, TestID: s.id, HostLabel: s.hostLabel, Workload: s.workload, RequestedContext: s.requestedContext, RequestedKVMode: s.requestedKVMode, Epoch: s.epoch})
		}(),
		PeakGPUVRAMBytesTotal: peakTotal,
		PeakGPUVRAMBytesByGPU: peakByGPU,
		AvgPromptTPS:          avgPrompt,
		AvgDecodeTPS:          avgDecode,
		MaxPromptTPS:          maxPrompt,
		MaxDecodeTPS:          maxDecode,
		ETAConfidence:         s.etaConfidence,
		StageDurations:        formatStageDurations(s.stageDurations),
		ProgressSamples:       s.progressSamples,
		LiveStatusEnabled:     s.displayEnabled,
		TelemetrySamplingSec:  telemetrySamplingSec,
	}
}

type liveManager struct {
	cfg      config
	suite    string
	total    int
	renderer *liveConsoleRenderer
	mu       sync.Mutex
	sessions map[string]*liveSession
}

func newLiveManager(cfg config, suite string, total int) *liveManager {
	m := &liveManager{
		cfg:      cfg,
		suite:    suite,
		total:    total,
		sessions: map[string]*liveSession{},
	}
	m.renderer = newLiveConsoleRenderer(cfg.LiveStatusMode, cfg.LiveDisplayInterval, suite, total, m)
	return m
}

func (m *liveManager) Close() {
	if m.renderer != nil {
		m.renderer.close()
	}
}

func (m *liveManager) register(s *liveSession) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessions[s.id] = s
}

func (m *liveManager) unregister(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, id)
}

func (m *liveManager) snapshot() []liveSample {
	m.mu.Lock()
	sessions := make([]*liveSession, 0, len(m.sessions))
	for _, s := range m.sessions {
		sessions = append(sessions, s)
	}
	m.mu.Unlock()
	sort.Slice(sessions, func(i, j int) bool { return sessions[i].testIndex < sessions[j].testIndex })
	out := make([]liveSample, 0, len(sessions))
	for _, s := range sessions {
		s.mu.Lock()
		out = append(out, s.lastSample)
		s.mu.Unlock()
	}
	return out
}

func telemetryPathFor(base string, opts liveSessionOptions) string {
	stamp := time.Now().UTC().Format("20060102_150405")
	name := fmt.Sprintf("%s_%s_%s_ctx%d_%s_e%d.telemetry.jsonl",
		stamp,
		sanitizeFileName(firstNonEmpty(opts.HostLabel, "host")),
		sanitizeFileName(firstNonEmpty(opts.Workload, "workload")),
		opts.RequestedContext,
		sanitizeFileName(firstNonEmpty(opts.RequestedKVMode, "kv")),
		max(1, opts.Epoch),
	)
	return filepath.Join(base, sanitizeFileName(firstNonEmpty(opts.Suite, "suite")), name)
}

func sanitizeFileName(v string) string {
	replacer := strings.NewReplacer("/", "_", "\\", "_", ":", "_", " ", "_")
	return replacer.Replace(v)
}

func ensureDir(path string) error {
	if path == "" || path == "." {
		return nil
	}
	return os.MkdirAll(path, 0o755)
}

func formatStageDurations(m map[string]time.Duration) string {
	if len(m) == 0 {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%s", k, m[k].Round(100*time.Millisecond)))
	}
	return strings.Join(parts, ";")
}

func cloneInt64Map(in map[string]int64) map[string]int64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]int64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func cloneFloat64Map(in map[string]float64) map[string]float64 {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]float64, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func formatInt64Map(in map[string]int64, prefix string) string {
	if len(in) == 0 {
		return ""
	}
	keys := make([]string, 0, len(in))
	for k := range in {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		label := k
		if prefix != "" && !strings.HasPrefix(label, prefix) {
			label = prefix + k
		}
		parts = append(parts, fmt.Sprintf("%s=%d", label, in[k]))
	}
	return strings.Join(parts, ";")
}
