package main

import (
	"io"
	"net/url"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

type workloadName string

const (
	workloadPrefillHeavy     workloadName = "prefill-heavy"
	workloadDecodeGrowth     workloadName = "decode-growth"
	workloadParallelAmplifier workloadName = "parallel-amplifier"
	workloadNearOOMStaircase workloadName = "near-oom-staircase"
)

type hostTarget struct {
	BaseURL       string
	Label         string
	URL           *url.URL
	Client        *api.Client
	KVSupportMode hostKVSupportMode
}

type config struct {
	Hosts            []hostTarget
	Model            string
	KVModes          []string
	Profile          string
	Workloads        []workloadName
	NumCtx           []int
	Concurrency      []int
	PromptTokens     int
	MaxTokens        int
	Warmup           int
	Epochs           int
	KeepAlive        time.Duration
	Seed             int
	Temperature      float64
	Timeout          time.Duration
	Stream           bool
	FailFast         bool
	ProbeInterval    time.Duration
	CaptureOllamaPS  bool
	CaptureGPU       bool
	CaptureRunnerRSS bool
	OutputPath       string
	JSONLPath        string
	SummaryPath      string
	OutputDir        string
	Debug            bool
	ProgressMode     progressMode
	ProgressWidth    int
}

type workloadSpec struct {
	Name               workloadName
	NumCtx             int
	PromptTokensTarget int
	MaxTokens          int
	Concurrency        int
}

type sweepCell struct {
	Host     hostTarget
	KVMode   string
	Workload workloadSpec
}

type hostPreflight struct {
	Version               string
	ModelFamily           string
	ModelQuant            string
	Reachable             bool
	OllamaPSBefore        string
	OllamaPSAfter         string
	KVSupport             map[string]kvSupportResult
}

type kvSupportResult struct {
	Supported bool
	Status    resultStatus
	Error     string
}

type kvRequestDisposition struct {
	Supported         bool
	RequestedOverride bool
	Error             string
}

type resultStatus string

const (
	statusOK          resultStatus = "ok"
	statusUnsupported resultStatus = "unsupported"
	statusFailed      resultStatus = "failed"
)

type promptCalibration struct {
	WordCount   int
	ActualTokens int
}

type promptKey struct {
	Host   string
	Model  string
	Target int
}

type workerResult struct {
	Host                string        `json:"host"`
	HostLabel           string        `json:"host_label"`
	ServerVersion       string        `json:"server_version"`
	Model               string        `json:"model"`
	Quant               string        `json:"quant"`
	KVModeRequested     string        `json:"kv_mode_requested"`
	KVModeResolved      string        `json:"kv_mode_resolved,omitempty"`
	KVAlgoResolved      string        `json:"kv_algo_resolved,omitempty"`
	KVBackendRequested  string        `json:"kv_backend_requested,omitempty"`
	KVPath              string        `json:"kv_path,omitempty"`
	Workload            string        `json:"workload"`
	NumCtx              int           `json:"num_ctx"`
	PromptTokensTarget  int           `json:"prompt_tokens_target"`
	PromptEvalCount     int           `json:"prompt_eval_count"`
	MaxTokens           int           `json:"max_tokens"`
	EvalCount           int           `json:"eval_count"`
	GeneratedTokens     int           `json:"generated_tokens"`
	LiveKVTokensTotal   int           `json:"live_kv_tokens_total"`
	CtxXConc            int           `json:"ctx_x_conc"`
	Concurrency         int           `json:"concurrency"`
	WorkerIndex         int           `json:"worker_index"`
	Epoch               int           `json:"epoch"`
	Warmup              bool          `json:"warmup"`
	PrefillTPS          float64       `json:"prefill_tps"`
	DecodeTPS           float64       `json:"decode_tps"`
	TTFTMS              float64       `json:"ttft_ms"`
	PromptEvalMS        float64       `json:"prompt_eval_ms"`
	EvalMS              float64       `json:"eval_ms"`
	LoadMS              float64       `json:"load_ms"`
	TotalMS             float64       `json:"total_ms"`
	WallMS              float64       `json:"wall_ms"`
	PeakVRAMBytes       *int64        `json:"peak_vram_bytes"`
	AvgGPUUtil          *float64      `json:"avg_gpu_util"`
	PeakGPUUtil         *float64      `json:"peak_gpu_util"`
	GPUMetricsAvailable bool          `json:"gpu_metrics_available"`
	HostRAMUsedBytes    *int64        `json:"host_ram_used_bytes"`
	PeakHostRAMBytes    *int64        `json:"peak_host_ram_bytes"`
	HostMetricsAvailable bool         `json:"host_metrics_available"`
	FullGPUResidency    bool          `json:"full_gpu_residency"`
	GPUOffloadRegression bool         `json:"gpu_offload_regression"`
	ProcessorStateBefore string       `json:"processor_state_before"`
	ProcessorStateAfter  string       `json:"processor_state_after"`
	Spilled             bool          `json:"spilled"`
	RunnerRSSBytes      int64         `json:"runner_rss_bytes"`
	SizeBytes           int64         `json:"size_bytes"`
	SizeVRAMBytes       int64         `json:"size_vram_bytes"`
	ContextLength       int           `json:"context_length"`
	GPUResidency        string        `json:"gpu_residency"`
	Status              resultStatus  `json:"status"`
	Success             *bool         `json:"success"`
	Error               string        `json:"error,omitempty"`
	RecordedAt          time.Time     `json:"recorded_at"`
}

type epochAggregate struct {
	Host                string
	HostLabel           string
	ServerVersion       string
	Model               string
	Quant               string
	KVModeRequested     string
	KVModeResolved      string
	KVAlgoResolved      string
	KVBackendRequested  string
	KVPath              string
	Workload            string
	NumCtx              int
	PromptTokensTarget  int
	PromptEvalCount     int
	MaxTokens           int
	EvalCount           int
	GeneratedTokens     int
	LiveKVTokensTotal   int
	CtxXConc            int
	Concurrency         int
	Epoch               int
	Warmup              bool
	PrefillTPS          float64
	DecodeTPS           float64
	TTFTMSMean          float64
	TTFTMSP95           float64
	LoadMS              float64
	TotalMS             float64
	WallMS              float64
	PeakVRAMBytes       *int64
	AvgGPUUtil          *float64
	PeakGPUUtil         *float64
	GPUMetricsAvailable bool
	HostRAMUsedBytes    *int64
	PeakHostRAMBytes    *int64
	HostMetricsAvailable bool
	FullGPUResidency    bool
	GPUOffloadRegression bool
	ProcessorStateBefore string
	ProcessorStateAfter  string
	Spilled             bool
	RunnerRSSBytes      int64
	Status              resultStatus
	Success             *bool
	Error               string
}

type staircaseRecord struct {
	HostLabel         string
	Host              string
	KVMode            string
	MaxFullGPUCtx     int
	LastStableNumCtx  int
	LastStableConc    int
	MaxStableCtxXConcurrency int
	LastFullGPUCtxXConcurrency int
	MaxLiveKVTokensTotal int
	FirstSpillCtx     int
	FirstSpillCtxXConcurrency int
	FailedNumCtx      int
	FailedConc        int
	FirstFailureCtxXConcurrency int
	FailureReason     string
}

type gpuStats struct {
	Available       bool
	PeakVRAMBytes   *int64
	AvgGPUUtil      *float64
	PeakGPUUtil     *float64
	ProcessVRAMBytes *int64
	SampleCount     int
}

type hostMemoryStats struct {
	Available         bool
	HostRAMUsedBytes  *int64
	PeakHostRAMBytes  *int64
	ProcessRSSBytes   *int64
	SampleCount       int
}

type progressMode string

type hostKVSupportMode string

const (
	hostKVSupportLegacy  hostKVSupportMode = "legacy"
	hostKVSupportRequest hostKVSupportMode = "request"
)

const (
	progressAuto progressMode = "auto"
	progressOn   progressMode = "on"
	progressOff  progressMode = "off"
)

type progressStep struct {
	Phase       string
	HostLabel   string
	KVMode      string
	Workload    string
	NumCtx      int
	Concurrency int
	Epoch       int
	EpochTotal  int
	Warmup      bool
}

type progressTracker struct {
	out       io.Writer
	enabled   bool
	debug     bool
	width     int
	total     int
	completed int
	startedAt time.Time
	current   progressStep
	lastWidth int
	pulseIndex int
	ticker    *time.Ticker
	stopCh    chan struct{}
	mu        sync.Mutex
	closed    bool
}
