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
	workloadPrefillHeavy      workloadName = "prefill-heavy"
	workloadDecodeGrowth      workloadName = "decode-growth"
	workloadParallelAmplifier workloadName = "parallel-amplifier"
	workloadNearOOMStaircase  workloadName = "near-oom-staircase"
	workloadFitCeiling        workloadName = "fit-ceiling"
	workloadLongContextRecall workloadName = "long-context-recall"
	workloadLongJSONRetention workloadName = "long-json-retention"
	workloadPromptFileRegress workloadName = "prompt-file-regression"
	workloadDecodeCorruption  workloadName = "decode-corruption-guard"
	workloadAgenticStructured workloadName = "agentic-structured"
	workloadNIAHRetrieval     workloadName = "niah-retrieval"
)

type hostTarget struct {
	BaseURL       string
	Label         string
	URL           *url.URL
	Client        *api.Client
	KVSupportMode hostKVSupportMode
}

type config struct {
	Hosts               []hostTarget
	Model               string
	KVModes             []string
	TurboMode           string
	FAModes             []bool
	QJLKModes           []bool
	QJLVModes           []bool
	Repeats             int
	RecallDistances     []int
	ToolSuite           string
	Profile             string
	Workloads           []workloadName
	NumCtx              []int
	Concurrency         []int
	PromptTokens        int
	MaxTokens           int
	Warmup              int
	Epochs              int
	KeepAlive           time.Duration
	Seed                int
	Temperature         float64
	Timeout             time.Duration
	Stream              bool
	FailFast            bool
	ProbeInterval       time.Duration
	CaptureOllamaPS     bool
	CaptureGPU          bool
	CaptureRunnerRSS    bool
	OutputPath          string
	JSONLPath           string
	SummaryPath         string
	MarkdownPath        string
	OutputDir           string
	BaselineHost        string
	TurboHost           string
	ModelFamilyOverride string
	ModelSizeLabel      string
	ResidualTailTokens  int
	ContextLadder       []int
	StretchContext      int
	MinFitDecode        int
	PromptFile          string
	LongJSONBytes       int
	TargetNativeCtx     int
	TargetYarnCtx       int
	Debug               bool
	ProgressMode        progressMode
	ProgressWidth       int
	LiveStatusMode      progressMode
	LiveTelemetry       bool
	LiveSampleInterval  time.Duration
	LiveDisplayInterval time.Duration
	LiveWriteInterval   time.Duration
	TelemetryDir        string
}

type workloadSpec struct {
	Name               workloadName
	NumCtx             int
	PromptTokensTarget int
	MaxTokens          int
	Concurrency        int
}

type sweepCell struct {
	Host               hostTarget
	KVMode             string
	Workload           workloadSpec
	FARequested        bool
	QJLKRequested      bool
	QJLVRequested      bool
	ResidualTailTokens int
}

type hostPreflight struct {
	Version        string
	ModelFamily    string
	ModelQuant     string
	Reachable      bool
	OllamaPSBefore string
	OllamaPSAfter  string
	KVSupport      map[string]kvSupportResult
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
	WordCount    int
	ActualTokens int
}

type promptKey struct {
	Host   string
	Model  string
	Target int
}

type workerResult struct {
	Host                            string       `json:"host"`
	HostLabel                       string       `json:"host_label"`
	ServerVersion                   string       `json:"server_version"`
	Model                           string       `json:"model"`
	ModelFamily                     string       `json:"model_family,omitempty"`
	ModelArch                       string       `json:"model_arch,omitempty"`
	ModelSizeLabel                  string       `json:"model_size_label,omitempty"`
	Quant                           string       `json:"quant"`
	KVModeRequested                 string       `json:"kv_mode_requested"`
	KVModeResolved                  string       `json:"kv_mode_resolved,omitempty"`
	KVModeRequestedK                string       `json:"kv_mode_requested_k,omitempty"`
	KVModeRequestedV                string       `json:"kv_mode_requested_v,omitempty"`
	KVModeResolvedK                 string       `json:"kv_mode_resolved_k,omitempty"`
	KVModeResolvedV                 string       `json:"kv_mode_resolved_v,omitempty"`
	RequestedCacheTypeK             string       `json:"requested_cache_type_k,omitempty"`
	RequestedCacheTypeV             string       `json:"requested_cache_type_v,omitempty"`
	EffectiveCacheTypeK             string       `json:"effective_cache_type_k,omitempty"`
	EffectiveCacheTypeV             string       `json:"effective_cache_type_v,omitempty"`
	RequestedMode                   string       `json:"requested_mode,omitempty"`
	EffectiveMode                   string       `json:"effective_mode,omitempty"`
	KVAlgoResolved                  string       `json:"kv_algo_resolved,omitempty"`
	KVAlgoResolvedK                 string       `json:"kv_algo_resolved_k,omitempty"`
	KVAlgoResolvedV                 string       `json:"kv_algo_resolved_v,omitempty"`
	KVBackendRequested              string       `json:"kv_backend_requested,omitempty"`
	KVPath                          string       `json:"kv_path,omitempty"`
	KVPathK                         string       `json:"kv_path_k,omitempty"`
	KVPathV                         string       `json:"kv_path_v,omitempty"`
	KVSymmetric                     bool         `json:"kv_symmetric"`
	KVAsymmetric                    bool         `json:"kv_asymmetric"`
	SymmetricRequested              bool         `json:"symmetric_requested"`
	SymmetricEffective              bool         `json:"symmetric_effective"`
	FallbackApplied                 bool         `json:"fallback_applied"`
	KOnlyFallback                   bool         `json:"k_only_fallback"`
	FallbackReason                  string       `json:"fallback_reason,omitempty"`
	TurboQuantPathKind              string       `json:"turboquant_path_kind,omitempty"`
	PathKind                        string       `json:"path_kind,omitempty"`
	NativeTurboQuantActive          bool         `json:"native_turboquant_active"`
	ReferenceTurboQuantActive       bool         `json:"reference_turboquant_active"`
	FlashAttentionRequested         bool         `json:"flash_attention_requested"`
	FlashAttentionEffective         bool         `json:"flash_attention_effective"`
	FAEnabled                       bool         `json:"fa_enabled"`
	FARequiredForVTurbo             bool         `json:"fa_required_for_v_turbo"`
	VTurboSupported                 bool         `json:"v_turbo_supported"`
	DetectedHeadDim                 int          `json:"detected_head_dim,omitempty"`
	ArchitectureClass               string       `json:"architecture_class,omitempty"`
	SupportTier                     string       `json:"support_tier,omitempty"`
	HybridKVArchitecture            bool         `json:"hybrid_kv_architecture"`
	TQBlockSize                     int          `json:"tq_block_size,omitempty"`
	VReconstructionComputeDType     string       `json:"v_reconstruction_compute_dtype,omitempty"`
	SegmentedHeadActive             bool         `json:"segmented_head_active"`
	SegmentedHeadPlan               string       `json:"segmented_head_plan,omitempty"`
	AttentionSurfacePolicyRequested string       `json:"attention_surface_policy_requested,omitempty"`
	AttentionSurfacePolicyEffective string       `json:"attention_surface_policy_effective,omitempty"`
	AttentionSurfacePolicy          string       `json:"attention_surface_policy,omitempty"`
	AttentionSurfaceBehavior        string       `json:"attention_surface_behavior,omitempty"`
	AttentionSurfaceOverrideApplied bool         `json:"attention_surface_override_applied"`
	AttentionSurfaceOverrideReason  string       `json:"attention_surface_override_reason,omitempty"`
	AttentionSurfaceClasses         string       `json:"attention_surface_classes,omitempty"`
	QJLKRequested                   bool         `json:"qjl_k_requested"`
	QJLKEffective                   bool         `json:"qjl_k_effective"`
	QJLVRequested                   bool         `json:"qjl_v_requested"`
	QJLVEffective                   bool         `json:"qjl_v_effective"`
	QJLKEnabled                     bool         `json:"qjl_k_enabled"`
	QJLVEnabled                     bool         `json:"qjl_v_enabled"`
	ResidualTailTokens              int          `json:"residual_tail_tokens,omitempty"`
	ExperimentalWeightQuantization  string       `json:"experimental_weight_quantization,omitempty"`
	ExperimentalWeightQuantPolicy   string       `json:"experimental_weight_quant_policy,omitempty"`
	ExperimentalWeightQuantSource   string       `json:"experimental_weight_quant_source,omitempty"`
	ExperimentalWeightQuantActive   bool         `json:"experimental_weight_quant_active"`
	GPUStatsSource                  string       `json:"gpu_stats_source,omitempty"`
	HostStatsSource                 string       `json:"host_stats_source,omitempty"`
	ValidationKind                  string       `json:"validation_kind,omitempty"`
	ValidationStatus                string       `json:"validation_status,omitempty"`
	ValidationObserved              string       `json:"validation_observed,omitempty"`
	ValidationExpected              string       `json:"validation_expected,omitempty"`
	ValidationError                 string       `json:"validation_error,omitempty"`
	RequestedNumCtx                 int          `json:"requested_num_ctx,omitempty"`
	AttemptedNumCtx                 int          `json:"attempted_num_ctx,omitempty"`
	EffectiveNumCtx                 int          `json:"effective_num_ctx,omitempty"`
	ContextRequested                int          `json:"context_requested,omitempty"`
	ContextEffective                int          `json:"context_effective,omitempty"`
	RequestedContextTopRung         int          `json:"requested_context_top_rung,omitempty"`
	ContextLadderIndex              int          `json:"context_ladder_index,omitempty"`
	ContextFallbackReason           string       `json:"context_fallback_reason,omitempty"`
	ContextFallbackDetail           string       `json:"context_fallback_detail,omitempty"`
	ContextFallbackStage            string       `json:"context_fallback_stage,omitempty"`
	ContextFallbackClass            string       `json:"context_fallback_class,omitempty"`
	LadderRejectedRungs             string       `json:"ladder_rejected_rungs,omitempty"`
	ModelFileSizeBytes              *int64       `json:"model_file_size_bytes,omitempty"`
	EstimatedKVFootprintBytes       *int64       `json:"estimated_kv_footprint_bytes,omitempty"`
	KVBufferBytesEstimate           *int64       `json:"kv_buffer_bytes_estimate,omitempty"`
	VisibleGPUCount                 int          `json:"visible_gpu_count,omitempty"`
	PerGPUVRAMGiB                   string       `json:"per_gpu_vram_gib,omitempty"`
	TotalVisibleVRAMBytes           *int64       `json:"total_visible_vram_bytes,omitempty"`
	ProcessVRAMBytes                *int64       `json:"process_vram_bytes,omitempty"`
	GPUVRAMUsedBytes                *int64       `json:"gpu_vram_used_bytes,omitempty"`
	GPUVRAMFreeBytes                *int64       `json:"gpu_vram_free_bytes,omitempty"`
	PeakHostRAMDeltaBytes           *int64       `json:"peak_host_ram_delta_bytes,omitempty"`
	HostRAMBeforeBytes              *int64       `json:"host_ram_before_bytes,omitempty"`
	HostRAMAfterLoadBytes           *int64       `json:"host_ram_after_load_bytes,omitempty"`
	HostRAMAfterPrefillBytes        *int64       `json:"host_ram_after_prefill_bytes,omitempty"`
	HostRAMAfterDecodeBytes         *int64       `json:"host_ram_after_decode_bytes,omitempty"`
	UsedHostAssist                  bool         `json:"used_host_assist"`
	UsedMMap                        bool         `json:"used_mmap"`
	UsedCPUAssist                   bool         `json:"used_cpu_assist"`
	LongContextCapSource            string       `json:"long_context_cap_source,omitempty"`
	NativeContextAdvertised         int          `json:"native_context_advertised,omitempty"`
	YarnContextAdvertised           int          `json:"yarn_context_advertised,omitempty"`
	ValidationCorruptionMarks       string       `json:"validation_corruption_markers,omitempty"`
	FitStatus                       string       `json:"fit_status,omitempty"`
	CorruptionStatus                string       `json:"corruption_status,omitempty"`
	CorruptionClass                 string       `json:"corruption_class,omitempty"`
	CorrectnessStatus               string       `json:"correctness_status,omitempty"`
	KLDivergenceVsBaseline          *float64     `json:"kl_divergence_vs_baseline,omitempty"`
	NIAHDepth                       int          `json:"niah_depth,omitempty"`
	NIAHPass                        bool         `json:"niah_pass"`
	EmptyOutput                     bool         `json:"empty_output"`
	TruncationDetected              bool         `json:"truncation_detected"`
	ResidencyKind                   string       `json:"residency_kind,omitempty"`
	Notes                           string       `json:"notes,omitempty"`
	TelemetryPathJSONL              string       `json:"telemetry_path_jsonl,omitempty"`
	PeakGPUVRAMBytesTotal           *int64       `json:"peak_gpu_vram_bytes_total,omitempty"`
	PeakGPUVRAMBytesByGPU           string       `json:"peak_gpu_vram_bytes_by_gpu,omitempty"`
	AvgPromptTPS                    *float64     `json:"avg_prompt_tps,omitempty"`
	AvgDecodeTPS                    *float64     `json:"avg_decode_tps,omitempty"`
	MaxPromptTPS                    *float64     `json:"max_prompt_tps,omitempty"`
	MaxDecodeTPS                    *float64     `json:"max_decode_tps,omitempty"`
	ETAConfidence                   string       `json:"eta_confidence,omitempty"`
	StageDurations                  string       `json:"stage_durations,omitempty"`
	ProgressSamples                 int          `json:"progress_samples,omitempty"`
	LiveStatusEnabled               bool         `json:"live_status_enabled"`
	TelemetrySamplingIntervalSec    int          `json:"telemetry_sampling_interval_sec,omitempty"`
	PromptTPSDeltaVsBaseline        *float64     `json:"prompt_tps_delta_vs_baseline,omitempty"`
	DecodeTPSDeltaVsBaseline        *float64     `json:"decode_tps_delta_vs_baseline,omitempty"`
	HostRAMDeltaVsBaseline          *int64       `json:"host_ram_delta_vs_baseline_bytes,omitempty"`
	GPUVRAMDeltaVsBaseline          *int64       `json:"gpu_vram_delta_vs_baseline_bytes,omitempty"`
	ContextDeltaVsBaseline          *int64       `json:"context_delta_vs_baseline,omitempty"`
	Workload                        string       `json:"workload"`
	NumCtx                          int          `json:"num_ctx"`
	PromptTokensTarget              int          `json:"prompt_tokens_target"`
	PromptEvalCount                 int          `json:"prompt_eval_count"`
	PromptTokens                    int          `json:"prompt_tokens,omitempty"`
	MaxTokens                       int          `json:"max_tokens"`
	EvalCount                       int          `json:"eval_count"`
	GeneratedTokens                 int          `json:"generated_tokens"`
	LiveKVTokensTotal               int          `json:"live_kv_tokens_total"`
	CtxXConc                        int          `json:"ctx_x_conc"`
	Concurrency                     int          `json:"concurrency"`
	WorkerIndex                     int          `json:"worker_index"`
	Epoch                           int          `json:"epoch"`
	Warmup                          bool         `json:"warmup"`
	PrefillTPS                      float64      `json:"prefill_tps"`
	DecodeTPS                       float64      `json:"decode_tps"`
	TTFTMS                          float64      `json:"ttft_ms"`
	PromptEvalMS                    float64      `json:"prompt_eval_ms"`
	EvalMS                          float64      `json:"eval_ms"`
	LoadMS                          float64      `json:"load_ms"`
	TotalMS                         float64      `json:"total_ms"`
	WallMS                          float64      `json:"wall_ms"`
	WallTimeS                       float64      `json:"wall_time_s,omitempty"`
	PeakVRAMBytes                   *int64       `json:"peak_vram_bytes"`
	AvgGPUUtil                      *float64     `json:"avg_gpu_util"`
	PeakGPUUtil                     *float64     `json:"peak_gpu_util"`
	GPUMetricsAvailable             bool         `json:"gpu_metrics_available"`
	HostRAMUsedBytes                *int64       `json:"host_ram_used_bytes"`
	PeakHostRAMBytes                *int64       `json:"peak_host_ram_bytes"`
	HostMetricsAvailable            bool         `json:"host_metrics_available"`
	FullGPUResidency                bool         `json:"full_gpu_residency"`
	GPUOffloadRegression            bool         `json:"gpu_offload_regression"`
	ProcessorStateBefore            string       `json:"processor_state_before"`
	ProcessorStateAfter             string       `json:"processor_state_after"`
	Spilled                         bool         `json:"spilled"`
	RunnerRSSBytes                  int64        `json:"runner_rss_bytes"`
	SizeBytes                       int64        `json:"size_bytes"`
	SizeVRAMBytes                   int64        `json:"size_vram_bytes"`
	ContextLength                   int          `json:"context_length"`
	GPUResidency                    string       `json:"gpu_residency"`
	Status                          resultStatus `json:"status"`
	Success                         *bool        `json:"success"`
	Error                           string       `json:"error,omitempty"`
	RecordedAt                      time.Time    `json:"recorded_at"`
}

type epochAggregate struct {
	Host                            string
	HostLabel                       string
	ServerVersion                   string
	Model                           string
	ModelFamily                     string
	ModelArch                       string
	ModelSizeLabel                  string
	Quant                           string
	KVModeRequested                 string
	KVModeResolved                  string
	KVModeRequestedK                string
	KVModeRequestedV                string
	KVModeResolvedK                 string
	KVModeResolvedV                 string
	RequestedCacheTypeK             string
	RequestedCacheTypeV             string
	EffectiveCacheTypeK             string
	EffectiveCacheTypeV             string
	RequestedMode                   string
	EffectiveMode                   string
	KVAlgoResolved                  string
	KVAlgoResolvedK                 string
	KVAlgoResolvedV                 string
	KVBackendRequested              string
	KVPath                          string
	KVPathK                         string
	KVPathV                         string
	KVSymmetric                     bool
	KVAsymmetric                    bool
	SymmetricRequested              bool
	SymmetricEffective              bool
	FallbackApplied                 bool
	KOnlyFallback                   bool
	FallbackReason                  string
	TurboQuantPathKind              string
	PathKind                        string
	NativeTurboQuantActive          bool
	ReferenceTurboQuantActive       bool
	FlashAttentionRequested         bool
	FlashAttentionEffective         bool
	FAEnabled                       bool
	FARequiredForVTurbo             bool
	VTurboSupported                 bool
	DetectedHeadDim                 int
	ArchitectureClass               string
	SupportTier                     string
	HybridKVArchitecture            bool
	TQBlockSize                     int
	VReconstructionComputeDType     string
	SegmentedHeadActive             bool
	SegmentedHeadPlan               string
	AttentionSurfacePolicyRequested string
	AttentionSurfacePolicyEffective string
	AttentionSurfacePolicy          string
	AttentionSurfaceBehavior        string
	AttentionSurfaceOverrideApplied bool
	AttentionSurfaceOverrideReason  string
	AttentionSurfaceClasses         string
	QJLKRequested                   bool
	QJLKEffective                   bool
	QJLVRequested                   bool
	QJLVEffective                   bool
	QJLKEnabled                     bool
	QJLVEnabled                     bool
	ResidualTailTokens              int
	ExperimentalWeightQuantization  string
	ExperimentalWeightQuantPolicy   string
	ExperimentalWeightQuantSource   string
	ExperimentalWeightQuantActive   bool
	GPUStatsSource                  string
	HostStatsSource                 string
	ValidationKind                  string
	ValidationStatus                string
	ValidationObserved              string
	ValidationExpected              string
	ValidationError                 string
	RequestedNumCtx                 int
	AttemptedNumCtx                 int
	EffectiveNumCtx                 int
	ContextRequested                int
	ContextEffective                int
	RequestedContextTopRung         int
	ContextLadderIndex              int
	ContextFallbackReason           string
	ContextFallbackDetail           string
	ContextFallbackStage            string
	ContextFallbackClass            string
	LadderRejectedRungs             string
	ModelFileSizeBytes              *int64
	EstimatedKVFootprintBytes       *int64
	KVBufferBytesEstimate           *int64
	VisibleGPUCount                 int
	PerGPUVRAMGiB                   string
	TotalVisibleVRAMBytes           *int64
	ProcessVRAMBytes                *int64
	GPUVRAMUsedBytes                *int64
	GPUVRAMFreeBytes                *int64
	PeakHostRAMDeltaBytes           *int64
	HostRAMBeforeBytes              *int64
	HostRAMAfterLoadBytes           *int64
	HostRAMAfterPrefillBytes        *int64
	HostRAMAfterDecodeBytes         *int64
	UsedHostAssist                  bool
	UsedMMap                        bool
	UsedCPUAssist                   bool
	LongContextCapSource            string
	NativeContextAdvertised         int
	YarnContextAdvertised           int
	ValidationCorruptionMarks       string
	FitStatus                       string
	CorruptionStatus                string
	CorruptionClass                 string
	CorrectnessStatus               string
	KLDivergenceVsBaseline          *float64
	NIAHDepth                       int
	NIAHPass                        bool
	EmptyOutput                     bool
	TruncationDetected              bool
	ResidencyKind                   string
	Notes                           string
	TelemetryPathJSONL              string
	PeakGPUVRAMBytesTotal           *int64
	PeakGPUVRAMBytesByGPU           string
	AvgPromptTPS                    *float64
	AvgDecodeTPS                    *float64
	MaxPromptTPS                    *float64
	MaxDecodeTPS                    *float64
	ETAConfidence                   string
	StageDurations                  string
	ProgressSamples                 int
	LiveStatusEnabled               bool
	TelemetrySamplingIntervalSec    int
	PromptTPSDeltaVsBaseline        *float64
	DecodeTPSDeltaVsBaseline        *float64
	HostRAMDeltaVsBaseline          *int64
	GPUVRAMDeltaVsBaseline          *int64
	ContextDeltaVsBaseline          *int64
	Workload                        string
	NumCtx                          int
	PromptTokensTarget              int
	PromptEvalCount                 int
	PromptTokens                    int
	MaxTokens                       int
	EvalCount                       int
	GeneratedTokens                 int
	LiveKVTokensTotal               int
	CtxXConc                        int
	Concurrency                     int
	Epoch                           int
	Warmup                          bool
	PrefillTPS                      float64
	DecodeTPS                       float64
	TTFTMSMean                      float64
	TTFTMSP95                       float64
	LoadMS                          float64
	TotalMS                         float64
	WallMS                          float64
	WallTimeS                       float64
	PeakVRAMBytes                   *int64
	AvgGPUUtil                      *float64
	PeakGPUUtil                     *float64
	GPUMetricsAvailable             bool
	HostRAMUsedBytes                *int64
	PeakHostRAMBytes                *int64
	HostMetricsAvailable            bool
	FullGPUResidency                bool
	GPUOffloadRegression            bool
	ProcessorStateBefore            string
	ProcessorStateAfter             string
	Spilled                         bool
	RunnerRSSBytes                  int64
	Status                          resultStatus
	Success                         *bool
	Error                           string
}

type staircaseRecord struct {
	HostLabel                   string
	Host                        string
	KVMode                      string
	MaxFullGPUCtx               int
	LastStableNumCtx            int
	LastStableConc              int
	MaxStableCtxXConcurrency    int
	LastFullGPUCtxXConcurrency  int
	MaxLiveKVTokensTotal        int
	FirstSpillCtx               int
	FirstSpillCtxXConcurrency   int
	FailedNumCtx                int
	FailedConc                  int
	FirstFailureCtxXConcurrency int
	FailureReason               string
}

type gpuStats struct {
	Available             bool
	Source                string
	PeakVRAMBytes         *int64
	CurrentVRAMBytes      *int64
	AvgGPUUtil            *float64
	PeakGPUUtil           *float64
	ProcessVRAMBytes      *int64
	TotalVisibleVRAMBytes *int64
	PerGPUUsedBytes       map[string]int64
	PerGPUFreeBytes       map[string]int64
	PeakPerGPUUsedBytes   map[string]int64
	PerGPUUtilPercent     map[string]float64
	PerGPUTempC           map[string]float64
	PerGPUPowerW          map[string]float64
	VisibleGPUCount       int
	SampleCount           int
}

type hostMemoryStats struct {
	Available             bool
	Source                string
	HostRAMBeforeBytes    *int64
	HostRAMUsedBytes      *int64
	PeakHostRAMBytes      *int64
	PeakHostRAMDeltaBytes *int64
	ProcessRSSBytes       *int64
	SelfRSSBytes          *int64
	SystemAvailableBytes  *int64
	CPUUtilPercent        *float64
	SampleCount           int
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
	out        io.Writer
	enabled    bool
	debug      bool
	width      int
	total      int
	completed  int
	startedAt  time.Time
	current    progressStep
	lastWidth  int
	pulseIndex int
	ticker     *time.Ticker
	stopCh     chan struct{}
	mu         sync.Mutex
	closed     bool
}
