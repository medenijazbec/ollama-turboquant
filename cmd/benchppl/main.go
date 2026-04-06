package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/tqbenchschema"
)

type hostKVSupportMode string

const (
	hostKVSupportLegacy  hostKVSupportMode = "legacy"
	hostKVSupportRequest hostKVSupportMode = "request"
)

type targetHost struct {
	BaseURL string
	Label   string
	Client  *api.Client
	Support hostKVSupportMode
}

type config struct {
	Hosts          []targetHost
	Model          string
	KVModes        []string
	FAModes        []bool
	QJLKModes      []bool
	QJLVModes      []bool
	CorpusPath     string
	ChunkTokens    int
	Chunks         int
	Seed           int
	OutputPath     string
	JSONLPath      string
	MarkdownPath   string
	OutputDir      string
	ModelSizeLabel string
	LiveStatus     string
	LiveTelemetry  bool
	LiveSampleSec  int
	TelemetryDir   string
}

type rawFlags struct {
	hosts          *string
	hostLabels     *string
	hostKVSupport  *string
	model          *string
	kvModes        *string
	faModes        *string
	qjlKModes      *string
	qjlVModes      *string
	corpus         *string
	chunkTokens    *int
	chunks         *int
	seed           *int
	output         *string
	jsonlOutput    *string
	markdownOutput *string
	summaryOutput  *string
	outputDir      *string
	modelSizeLabel *string
	liveStatus     *string
	liveTelemetry  *string
	liveSampleSec  *int
	telemetryDir   *string
}

type chunkResult struct {
	HostLabel               string        `json:"host_label"`
	Host                    string        `json:"host"`
	Model                   string        `json:"model"`
	ModelFamily             string        `json:"model_family,omitempty"`
	ModelArch               string        `json:"model_arch,omitempty"`
	ModelSizeLabel          string        `json:"model_size_label,omitempty"`
	RequestedCacheTypeK     string        `json:"requested_cache_type_k"`
	RequestedCacheTypeV     string        `json:"requested_cache_type_v"`
	EffectiveCacheTypeK     string        `json:"effective_cache_type_k,omitempty"`
	EffectiveCacheTypeV     string        `json:"effective_cache_type_v,omitempty"`
	SymmetricRequested      bool          `json:"symmetric_requested"`
	SymmetricEffective      bool          `json:"symmetric_effective"`
	FlashAttentionRequested bool          `json:"flash_attention_requested"`
	FlashAttentionEffective bool          `json:"flash_attention_effective"`
	QJLKRequested           bool          `json:"qjl_k_requested"`
	QJLKEffective           bool          `json:"qjl_k_effective"`
	QJLVRequested           bool          `json:"qjl_v_requested"`
	QJLVEffective           bool          `json:"qjl_v_effective"`
	PathKind                string        `json:"path_kind,omitempty"`
	ContextRequested        int           `json:"context_requested,omitempty"`
	ContextEffective        int           `json:"context_effective,omitempty"`
	PromptTokens            int           `json:"prompt_tokens,omitempty"`
	GeneratedTokens         int           `json:"generated_tokens,omitempty"`
	PromptTPS               float64       `json:"prompt_tps,omitempty"`
	DecodeTPS               float64       `json:"decode_tps,omitempty"`
	WallTimeS               float64       `json:"wall_time_s,omitempty"`
	KVBufferBytesEstimate   *int64        `json:"kv_buffer_bytes_estimate,omitempty"`
	GPUVRAMUsedBytes        *int64        `json:"gpu_vram_used_bytes,omitempty"`
	GPUVRAMFreeBytes        *int64        `json:"gpu_vram_free_bytes,omitempty"`
	HostRAMUsedBytes        *int64        `json:"host_ram_used_bytes,omitempty"`
	FitStatus               string        `json:"fit_status,omitempty"`
	CorruptionStatus        string        `json:"corruption_status,omitempty"`
	CorrectnessStatus       string        `json:"correctness_status,omitempty"`
	FallbackReason          string        `json:"fallback_reason,omitempty"`
	Notes                   string        `json:"notes,omitempty"`
	ResidencyKind           string        `json:"residency_kind,omitempty"`
	ChunkIndex              int           `json:"chunk_index"`
	TokenCount              int           `json:"token_count"`
	NegativeLogLikelihood   float64       `json:"negative_log_likelihood,omitempty"`
	Perplexity              float64       `json:"perplexity,omitempty"`
	PPLDeltaVsBaseline      *float64      `json:"ppl_delta_vs_baseline,omitempty"`
	KLDivergenceVsBaseline  *float64      `json:"kl_divergence_vs_baseline,omitempty"`
	Status                  string        `json:"status"`
	Error                   string        `json:"error,omitempty"`
	RecordedAt              time.Time     `json:"recorded_at"`
	PerTokenLogprobs        []api.Logprob `json:"-"`
	TelemetryPathJSONL      string        `json:"telemetry_path_jsonl,omitempty"`
	PeakHostRAMBytes        *int64        `json:"peak_host_ram_bytes,omitempty"`
	PeakGPUVRAMBytesTotal   *int64        `json:"peak_gpu_vram_bytes_total,omitempty"`
	PeakGPUVRAMBytesByGPU   string        `json:"peak_gpu_vram_bytes_by_gpu,omitempty"`
	AvgPromptTPS            *float64      `json:"avg_prompt_tps,omitempty"`
	AvgDecodeTPS            *float64      `json:"avg_decode_tps,omitempty"`
	MaxPromptTPS            *float64      `json:"max_prompt_tps,omitempty"`
	MaxDecodeTPS            *float64      `json:"max_decode_tps,omitempty"`
	ETAConfidence           string        `json:"eta_confidence,omitempty"`
	ProgressSamples         int           `json:"progress_samples,omitempty"`
	LiveStatusEnabled       bool          `json:"live_status_enabled,omitempty"`
	TelemetrySamplingSec    int           `json:"telemetry_sampling_interval_sec,omitempty"`
}

type aggregateResult struct {
	HostLabel               string
	Host                    string
	Model                   string
	ModelSizeLabel          string
	RequestedCacheTypeK     string
	RequestedCacheTypeV     string
	EffectiveCacheTypeK     string
	EffectiveCacheTypeV     string
	SymmetricRequested      bool
	SymmetricEffective      bool
	FlashAttentionRequested bool
	FlashAttentionEffective bool
	QJLKRequested           bool
	QJLKEffective           bool
	QJLVRequested           bool
	QJLVEffective           bool
	PathKind                string
	PromptTPS               float64
	DecodeTPS               float64
	WallTimeS               float64
	KVBufferBytesEstimate   *int64
	GPUVRAMUsedBytes        *int64
	GPUVRAMFreeBytes        *int64
	HostRAMUsedBytes        *int64
	FitStatus               string
	CorruptionStatus        string
	CorrectnessStatus       string
	FallbackReason          string
	Notes                   string
	ResidencyKind           string
	TokenCount              int
	NegativeLogLikelihood   float64
	Perplexity              float64
	PPLDeltaVsBaseline      *float64
	KLDivergenceVsBaseline  *float64
	Status                  string
	Error                   string
	ChunkCount              int
	TelemetryPathJSONL      string
	PeakHostRAMBytes        *int64
	PeakGPUVRAMBytesTotal   *int64
	PeakGPUVRAMBytesByGPU   string
	AvgPromptTPS            *float64
	AvgDecodeTPS            *float64
	MaxPromptTPS            *float64
	MaxDecodeTPS            *float64
	ETAConfidence           string
	ProgressSamples         int
	LiveStatusEnabled       bool
	TelemetrySamplingSec    int
}

func main() {
	flags := parseFlags()
	flag.Parse()

	cfg, err := loadConfig(flags)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	if err := ensureOutputDirs(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	chunks, err := loadCorpusChunks(cfg.CorpusPath, cfg.ChunkTokens, cfg.Chunks)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	rows, aggregates := runPPLBench(cfg, chunks)
	if err := writeJSONL(cfg.JSONLPath, rows); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR writing JSONL: %v\n", err)
		os.Exit(1)
	}
	if err := writeCSV(cfg.OutputPath, aggregates); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR writing CSV: %v\n", err)
		os.Exit(1)
	}
	if err := writeMarkdown(cfg.MarkdownPath, aggregates); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR writing markdown: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Wrote CSV summary to %s\n", cfg.OutputPath)
	fmt.Printf("Wrote JSONL rows to %s\n", cfg.JSONLPath)
	fmt.Printf("Wrote markdown summary to %s\n", cfg.MarkdownPath)
}

func parseFlags() rawFlags {
	return rawFlags{
		hosts:          flag.String("hosts", "", "Comma-separated Ollama hosts"),
		hostLabels:     flag.String("host-labels", "", "Comma-separated host labels"),
		hostKVSupport:  flag.String("host-kv-support", "", "Comma-separated host KV support modes [legacy|request]"),
		model:          flag.String("model", "", "Model to benchmark"),
		kvModes:        flag.String("kv-modes", "f16,q8_0,q4_0,tq25,tq35,q8_0/tq35", "Comma-separated KV modes"),
		faModes:        flag.String("fa-modes", "both", "Flash Attention modes [on|off|both]"),
		qjlKModes:      flag.String("qjl-k-modes", "off", "Experimental K-side QJL request modes [on|off|both]"),
		qjlVModes:      flag.String("qjl-v-modes", "off", "Experimental V-side QJL request modes [on|off|both]; blocked lanes remain explicit"),
		corpus:         flag.String("corpus", "", "Path to local corpus text"),
		chunkTokens:    flag.Int("chunk-tokens", 512, "Approximate word/token chunk size"),
		chunks:         flag.Int("chunks", 16, "Maximum number of chunks"),
		seed:           flag.Int("seed", 42, "Deterministic seed label"),
		output:         flag.String("output", "", "CSV output path"),
		jsonlOutput:    flag.String("jsonl-output", "", "JSONL output path"),
		markdownOutput: flag.String("markdown-output", "", "Markdown output path"),
		summaryOutput:  flag.String("summary-output", "", "Markdown output path alias"),
		outputDir:      flag.String("output-dir", "/results", "Output directory"),
		modelSizeLabel: flag.String("model-size-label", "", "Optional model size label override"),
		liveStatus:     flag.String("live-status", "auto", "Live console status [auto|on|off]"),
		liveTelemetry:  flag.String("live-telemetry", "on", "Live JSONL telemetry [on|off]"),
		liveSampleSec:  flag.Int("live-sample-interval-sec", 1, "Live telemetry sample interval seconds"),
		telemetryDir:   flag.String("telemetry-dir", "", "Per-test telemetry output directory"),
	}
}

func loadConfig(r rawFlags) (config, error) {
	if strings.TrimSpace(*r.model) == "" {
		return config{}, errors.New("--model is required")
	}
	if strings.TrimSpace(*r.corpus) == "" {
		return config{}, errors.New("--corpus is required")
	}

	hosts := splitCSV(*r.hosts)
	if len(hosts) == 0 {
		return config{}, errors.New("--hosts is required")
	}
	labels := splitCSV(*r.hostLabels)
	if len(labels) == 0 {
		labels = make([]string, len(hosts))
		for i := range hosts {
			labels[i] = fmt.Sprintf("host%d", i+1)
		}
	}
	if len(labels) != len(hosts) {
		return config{}, errors.New("--host-labels count must match --hosts count")
	}

	supportModes, err := parseSupportModes(*r.hostKVSupport, len(hosts))
	if err != nil {
		return config{}, err
	}

	cfg := config{
		Model:          strings.TrimSpace(*r.model),
		KVModes:        splitCSV(*r.kvModes),
		FAModes:        parseFAModes(*r.faModes),
		QJLKModes:      parseQJLModes(*r.qjlKModes),
		QJLVModes:      parseQJLModes(*r.qjlVModes),
		CorpusPath:     strings.TrimSpace(*r.corpus),
		ChunkTokens:    max(1, *r.chunkTokens),
		Chunks:         max(1, *r.chunks),
		Seed:           *r.seed,
		OutputDir:      strings.TrimSpace(*r.outputDir),
		ModelSizeLabel: strings.TrimSpace(*r.modelSizeLabel),
		LiveStatus:     "auto",
		LiveTelemetry:  true,
		LiveSampleSec:  1,
	}
	if r.liveStatus != nil {
		cfg.LiveStatus = strings.TrimSpace(*r.liveStatus)
	}
	if r.liveTelemetry != nil && strings.EqualFold(strings.TrimSpace(*r.liveTelemetry), "off") {
		cfg.LiveTelemetry = false
	}
	if r.liveSampleSec != nil {
		cfg.LiveSampleSec = max(1, *r.liveSampleSec)
	}

	for i, host := range hosts {
		u, err := url.Parse(host)
		if err != nil {
			return config{}, fmt.Errorf("invalid host %q: %w", host, err)
		}
		cfg.Hosts = append(cfg.Hosts, targetHost{
			BaseURL: host,
			Label:   labels[i],
			Client:  api.NewClient(u, http.DefaultClient),
			Support: supportModes[i],
		})
	}

	baseName := sanitizeFileName(cfg.Model) + "_ppl"
	cfg.OutputPath = firstNonEmpty(strings.TrimSpace(*r.output), filepath.Join(cfg.OutputDir, baseName+".csv"))
	cfg.JSONLPath = firstNonEmpty(strings.TrimSpace(*r.jsonlOutput), filepath.Join(cfg.OutputDir, baseName+".jsonl"))
	cfg.MarkdownPath = firstNonEmpty(strings.TrimSpace(*r.markdownOutput), strings.TrimSpace(*r.summaryOutput), filepath.Join(cfg.OutputDir, baseName+".md"))
	cfg.TelemetryDir = filepath.Join(cfg.OutputDir, "telemetry")
	if r.telemetryDir != nil && strings.TrimSpace(*r.telemetryDir) != "" {
		cfg.TelemetryDir = strings.TrimSpace(*r.telemetryDir)
	}

	return cfg, nil
}

func runPPLBench(cfg config, chunks []string) ([]chunkResult, []aggregateResult) {
	var rows []chunkResult
	total := countPPLCases(cfg, chunks)
	testIndex := 0
	for _, host := range cfg.Hosts {
		for _, mode := range cfg.KVModes {
			kType, vType, ok := splitKVMode(mode)
			if !ok {
				rows = append(rows, chunkResult{
					HostLabel:         host.Label,
					Host:              host.BaseURL,
					Model:             cfg.Model,
					ModelSizeLabel:    deriveSizeLabel(cfg.Model, cfg.ModelSizeLabel),
					FitStatus:         string(tqbenchschema.FitStatusUnsupported),
					CorruptionStatus:  string(tqbenchschema.CorruptionStatusSkipped),
					CorrectnessStatus: string(tqbenchschema.CorrectnessStatusUnsupported),
					ResidencyKind:     string(tqbenchschema.ResidencyUnknown),
					Status:            "unsupported",
					Error:             "invalid kv mode",
					RecordedAt:        time.Now().UTC(),
				})
				continue
			}
			for _, faRequested := range cfg.FAModes {
				for _, qjlKRequested := range cfg.QJLKModes {
					for _, qjlVRequested := range cfg.QJLVModes {
						if host.Support == hostKVSupportLegacy && (kType != "f16" || vType != "f16") {
							rows = append(rows, chunkResult{
								HostLabel:               host.Label,
								Host:                    host.BaseURL,
								Model:                   cfg.Model,
								ModelSizeLabel:          deriveSizeLabel(cfg.Model, cfg.ModelSizeLabel),
								RequestedCacheTypeK:     kType,
								RequestedCacheTypeV:     vType,
								SymmetricRequested:      strings.EqualFold(kType, vType),
								FlashAttentionRequested: faRequested,
								QJLKRequested:           qjlKRequested,
								QJLVRequested:           qjlVRequested,
								FitStatus:               string(tqbenchschema.FitStatusUnsupported),
								CorruptionStatus:        string(tqbenchschema.CorruptionStatusSkipped),
								CorrectnessStatus:       string(tqbenchschema.CorrectnessStatusUnsupported),
								ResidencyKind:           string(tqbenchschema.ResidencyUnknown),
								Notes:                   "legacy host does not accept kv_cache_type overrides",
								Status:                  "unsupported",
								Error:                   "unsupported kv_cache_type on target host",
								RecordedAt:              time.Now().UTC(),
							})
							continue
						}
						for idx, chunk := range chunks {
							testIndex++
							rows = append(rows, scoreChunk(cfg, host, kType, vType, faRequested, qjlKRequested, qjlVRequested, idx, len(chunks), testIndex, total, chunk))
						}
					}
				}
			}
		}
	}

	aggregates := aggregateChunkResults(rows)
	applyBaselinePPLDeltas(aggregates)
	applyBaselineKLD(rows, aggregates)
	return rows, aggregates
}

func scoreChunk(cfg config, host targetHost, kType string, vType string, faRequested bool, qjlKRequested bool, qjlVRequested bool, idx int, chunkTotal int, testIndex int, testTotal int, chunk string) chunkResult {
	row := chunkResult{
		HostLabel:               host.Label,
		Host:                    host.BaseURL,
		Model:                   cfg.Model,
		ModelSizeLabel:          deriveSizeLabel(cfg.Model, cfg.ModelSizeLabel),
		RequestedCacheTypeK:     kType,
		RequestedCacheTypeV:     vType,
		SymmetricRequested:      strings.EqualFold(kType, vType),
		FlashAttentionRequested: faRequested,
		QJLKRequested:           qjlKRequested,
		QJLVRequested:           qjlVRequested,
		FitStatus:               string(tqbenchschema.FitStatusFailed),
		CorruptionStatus:        string(tqbenchschema.CorruptionStatusPass),
		CorrectnessStatus:       string(tqbenchschema.CorrectnessStatusFail),
		ResidencyKind:           string(tqbenchschema.ResidencyUnknown),
		ChunkIndex:              idx,
		RecordedAt:              time.Now().UTC(),
		Notes:                   "chunk scoring uses raw prompt+target bench route",
	}
	live := newPPLLiveSession(cfg, host, row, idx+1, chunkTotal, testIndex, testTotal)
	defer func() {
		if live != nil && !live.closed {
			summary := live.Close("failed", row.Error, &row)
			applyPPLLiveSummary(&row, summary)
		}
	}()

	prompt, target, ok := splitChunkPromptTarget(chunk)
	if !ok {
		row.Status = "failed"
		row.Error = "chunk too small to split into prompt and target"
		row.Notes = "request_schema_failed"
		if live != nil {
			summary := live.Close("failed", row.Error, &row)
			applyPPLLiveSummary(&row, summary)
		}
		return row
	}

	options := map[string]any{
		"num_ctx":         max(4096, cfg.ChunkTokens*4),
		"seed":            cfg.Seed,
		"temperature":     0,
		"top_p":           1,
		"repeat_penalty":  1,
		"flash_attention": faRequested,
	}
	if strings.EqualFold(kType, vType) {
		options["kv_cache_type"] = kType
	} else {
		options["kv_cache_type_k"] = kType
		options["kv_cache_type_v"] = vType
	}
	if qjlKRequested {
		options["turboquant_qjl_k"] = true
	}
	if qjlVRequested {
		options["turboquant_qjl_v"] = true
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	start := time.Now()
	if live != nil {
		live.SetStage("scoring")
	}
	resp, err := host.Client.BenchScore(ctx, &api.BenchScoreRequest{
		Model:                cfg.Model,
		Prompt:               prompt,
		Target:               target,
		Options:              options,
		IncludeTokenLogprobs: true,
		TopLogprobs:          5,
	})
	row.WallTimeS = time.Since(start).Seconds()
	if err != nil {
		row.Status = classifyPPLStatus(err)
		row.Error = err.Error()
		if row.Status == "unsupported" {
			row.FitStatus = string(tqbenchschema.FitStatusUnsupported)
			row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusUnsupported)
		}
		if live != nil {
			summary := live.Close("failed", row.Error, &row)
			applyPPLLiveSummary(&row, summary)
		}
		return row
	}

	row.Status = "ok"
	row.FitStatus = string(tqbenchschema.FitStatusFit)
	row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusPass)
	row.ModelArch = resp.ArchitectureClass
	row.EffectiveCacheTypeK = firstNonEmpty(resp.ResolvedKVCacheTypeK, resp.KVCacheRequestedK, kType)
	row.EffectiveCacheTypeV = firstNonEmpty(resp.ResolvedKVCacheTypeV, resp.KVCacheRequestedV, vType)
	row.SymmetricEffective = strings.EqualFold(row.EffectiveCacheTypeK, row.EffectiveCacheTypeV)
	row.FlashAttentionEffective = resp.FAEnabled
	row.QJLKEffective = resp.QJLKEnabled
	row.QJLVEffective = resp.QJLVEnabled
	row.PathKind = resp.TurboQuantPathKind
	row.ContextRequested = options["num_ctx"].(int)
	row.ContextEffective = row.ContextRequested
	row.PromptTokens = resp.PromptEvalCount
	row.GeneratedTokens = resp.EvalCount
	if resp.PromptEvalDuration > 0 && resp.PromptEvalCount > 0 {
		row.PromptTPS = float64(resp.PromptEvalCount) / resp.PromptEvalDuration.Seconds()
	}
	if resp.EvalDuration > 0 && resp.EvalCount > 0 {
		row.DecodeTPS = float64(resp.EvalCount) / resp.EvalDuration.Seconds()
	}
	if resp.KVCacheBytes > 0 {
		row.KVBufferBytesEstimate = int64Ptr(int64(resp.KVCacheBytes))
	}
	if resp.TotalVRAMBytes > 0 {
		row.GPUVRAMUsedBytes = int64Ptr(int64(resp.TotalVRAMBytes))
	}
	row.FallbackReason = resp.FallbackReason
	row.TokenCount = resp.TokenCount
	row.NegativeLogLikelihood = resp.NegativeLogLikelihood
	row.Perplexity = resp.Perplexity
	row.PerTokenLogprobs = resp.PerTokenLogprobs
	if live != nil {
		live.SetEffective(row.EffectiveCacheTypeK, row.EffectiveCacheTypeV, row.FlashAttentionEffective, row.QJLKEffective, row.QJLVEffective, row.FallbackReason, row.PromptTPS, row.DecodeTPS)
		live.SetStage("save")
		summary := live.Close("done", "", &row)
		applyPPLLiveSummary(&row, summary)
	}

	return row
}

func aggregateChunkResults(rows []chunkResult) []aggregateResult {
	type key struct {
		hostLabel string
		host      string
		model     string
		reqK      string
		reqV      string
		fa        bool
		qjlK      bool
		qjlV      bool
	}
	aggMap := map[key]*aggregateResult{}
	for _, row := range rows {
		k := key{hostLabel: row.HostLabel, host: row.Host, model: row.Model, reqK: row.RequestedCacheTypeK, reqV: row.RequestedCacheTypeV, fa: row.FlashAttentionRequested, qjlK: row.QJLKRequested, qjlV: row.QJLVRequested}
		if aggMap[k] == nil {
			aggMap[k] = &aggregateResult{
				HostLabel:               row.HostLabel,
				Host:                    row.Host,
				Model:                   row.Model,
				ModelSizeLabel:          row.ModelSizeLabel,
				RequestedCacheTypeK:     row.RequestedCacheTypeK,
				RequestedCacheTypeV:     row.RequestedCacheTypeV,
				FlashAttentionRequested: row.FlashAttentionRequested,
				QJLKRequested:           row.QJLKRequested,
				QJLVRequested:           row.QJLVRequested,
				FitStatus:               row.FitStatus,
				CorruptionStatus:        row.CorruptionStatus,
				CorrectnessStatus:       row.CorrectnessStatus,
				ResidencyKind:           row.ResidencyKind,
				Status:                  row.Status,
				Error:                   row.Error,
			}
		}
		cur := aggMap[k]
		cur.ChunkCount++
		cur.TokenCount += row.TokenCount
		cur.NegativeLogLikelihood += row.NegativeLogLikelihood
		cur.PromptTPS += row.PromptTPS
		cur.DecodeTPS += row.DecodeTPS
		cur.WallTimeS += row.WallTimeS
		cur.EffectiveCacheTypeK = firstNonEmpty(cur.EffectiveCacheTypeK, row.EffectiveCacheTypeK)
		cur.EffectiveCacheTypeV = firstNonEmpty(cur.EffectiveCacheTypeV, row.EffectiveCacheTypeV)
		cur.SymmetricRequested = row.SymmetricRequested
		cur.SymmetricEffective = row.SymmetricEffective
		cur.FlashAttentionEffective = row.FlashAttentionEffective
		cur.QJLKEffective = cur.QJLKEffective || row.QJLKEffective
		cur.QJLVEffective = cur.QJLVEffective || row.QJLVEffective
		cur.PathKind = firstNonEmpty(cur.PathKind, row.PathKind)
		cur.FallbackReason = firstNonEmpty(cur.FallbackReason, row.FallbackReason)
		cur.Notes = firstNonEmpty(cur.Notes, row.Notes)
		cur.TelemetryPathJSONL = firstNonEmpty(cur.TelemetryPathJSONL, row.TelemetryPathJSONL)
		cur.KVBufferBytesEstimate = maxInt64Ptr(cur.KVBufferBytesEstimate, row.KVBufferBytesEstimate)
		cur.GPUVRAMUsedBytes = maxInt64Ptr(cur.GPUVRAMUsedBytes, row.GPUVRAMUsedBytes)
		cur.GPUVRAMFreeBytes = maxInt64Ptr(cur.GPUVRAMFreeBytes, row.GPUVRAMFreeBytes)
		cur.HostRAMUsedBytes = maxInt64Ptr(cur.HostRAMUsedBytes, row.HostRAMUsedBytes)
		cur.PeakHostRAMBytes = maxInt64Ptr(cur.PeakHostRAMBytes, row.PeakHostRAMBytes)
		cur.PeakGPUVRAMBytesTotal = maxInt64Ptr(cur.PeakGPUVRAMBytesTotal, row.PeakGPUVRAMBytesTotal)
		cur.PeakGPUVRAMBytesByGPU = firstNonEmpty(cur.PeakGPUVRAMBytesByGPU, row.PeakGPUVRAMBytesByGPU)
		cur.AvgPromptTPS = chooseFloatPtr(cur.AvgPromptTPS, row.AvgPromptTPS)
		cur.AvgDecodeTPS = chooseFloatPtr(cur.AvgDecodeTPS, row.AvgDecodeTPS)
		cur.MaxPromptTPS = chooseFloatPtr(cur.MaxPromptTPS, row.MaxPromptTPS)
		cur.MaxDecodeTPS = chooseFloatPtr(cur.MaxDecodeTPS, row.MaxDecodeTPS)
		cur.ETAConfidence = firstNonEmpty(cur.ETAConfidence, row.ETAConfidence)
		cur.ProgressSamples += row.ProgressSamples
		cur.LiveStatusEnabled = cur.LiveStatusEnabled || row.LiveStatusEnabled
		if row.TelemetrySamplingSec > cur.TelemetrySamplingSec {
			cur.TelemetrySamplingSec = row.TelemetrySamplingSec
		}
		if row.Status != "ok" {
			cur.Status = row.Status
			if cur.Error == "" {
				cur.Error = row.Error
			}
		}
		if row.CorrectnessStatus == string(tqbenchschema.CorrectnessStatusFail) {
			cur.CorrectnessStatus = row.CorrectnessStatus
		}
		if row.CorruptionStatus == string(tqbenchschema.CorruptionStatusFail) {
			cur.CorruptionStatus = row.CorruptionStatus
		}
		if row.KLDivergenceVsBaseline != nil {
			if cur.KLDivergenceVsBaseline == nil {
				cur.KLDivergenceVsBaseline = new(float64)
			}
			*cur.KLDivergenceVsBaseline += *row.KLDivergenceVsBaseline
		}
	}

	out := make([]aggregateResult, 0, len(aggMap))
	for _, agg := range aggMap {
		if agg.ChunkCount > 0 {
			agg.PromptTPS /= float64(agg.ChunkCount)
			agg.DecodeTPS /= float64(agg.ChunkCount)
			if agg.KLDivergenceVsBaseline != nil {
				*agg.KLDivergenceVsBaseline /= float64(agg.ChunkCount)
			}
		}
		if agg.TokenCount > 0 {
			agg.Perplexity = math.Exp(agg.NegativeLogLikelihood / float64(agg.TokenCount))
		}
		out = append(out, *agg)
	}
	slices.SortFunc(out, func(a, b aggregateResult) int {
		if a.HostLabel != b.HostLabel {
			return strings.Compare(a.HostLabel, b.HostLabel)
		}
		if a.Model != b.Model {
			return strings.Compare(a.Model, b.Model)
		}
		if a.RequestedCacheTypeK != b.RequestedCacheTypeK {
			return strings.Compare(a.RequestedCacheTypeK, b.RequestedCacheTypeK)
		}
		if a.RequestedCacheTypeV != b.RequestedCacheTypeV {
			return strings.Compare(a.RequestedCacheTypeV, b.RequestedCacheTypeV)
		}
		if a.QJLKRequested != b.QJLKRequested {
			if !a.QJLKRequested {
				return -1
			}
			return 1
		}
		if a.QJLVRequested != b.QJLVRequested {
			if !a.QJLVRequested {
				return -1
			}
			return 1
		}
		if a.FlashAttentionRequested == b.FlashAttentionRequested {
			return 0
		}
		if !a.FlashAttentionRequested {
			return -1
		}
		return 1
	})
	return out
}

func applyBaselinePPLDeltas(rows []aggregateResult) {
	type baseKey struct {
		model string
		fa    bool
	}
	baseline := map[baseKey]aggregateResult{}
	for _, row := range rows {
		if row.HostLabel == "baseline" && row.RequestedCacheTypeK == "f16" && row.RequestedCacheTypeV == "f16" && row.Status == "ok" {
			baseline[baseKey{model: row.Model, fa: row.FlashAttentionRequested}] = row
		}
	}
	for i := range rows {
		base, ok := baseline[baseKey{model: rows[i].Model, fa: rows[i].FlashAttentionRequested}]
		if !ok || base.Perplexity == 0 || rows[i].Status != "ok" {
			continue
		}
		delta := rows[i].Perplexity - base.Perplexity
		rows[i].PPLDeltaVsBaseline = &delta
	}
}

func applyBaselineKLD(rows []chunkResult, aggregates []aggregateResult) {
	type chunkKey struct {
		model string
		fa    bool
		index int
		qjlK  bool
		qjlV  bool
	}
	baseline := map[chunkKey]chunkResult{}
	for _, row := range rows {
		if row.HostLabel == "baseline" && row.RequestedCacheTypeK == "f16" && row.RequestedCacheTypeV == "f16" && row.Status == "ok" {
			baseline[chunkKey{model: row.Model, fa: row.FlashAttentionRequested, index: row.ChunkIndex, qjlK: row.QJLKRequested, qjlV: row.QJLVRequested}] = row
		}
	}
	for i := range rows {
		base, ok := baseline[chunkKey{model: rows[i].Model, fa: rows[i].FlashAttentionRequested, index: rows[i].ChunkIndex, qjlK: rows[i].QJLKRequested, qjlV: rows[i].QJLVRequested}]
		if !ok || rows[i].Status != "ok" {
			continue
		}
		if kld, ok := computeSparseKLD(base.PerTokenLogprobs, rows[i].PerTokenLogprobs); ok {
			rows[i].KLDivergenceVsBaseline = &kld
		} else if rows[i].Notes == "" {
			rows[i].Notes = "kld_unavailable_sparse_support"
		} else if !strings.Contains(rows[i].Notes, "kld_unavailable_sparse_support") {
			rows[i].Notes += " | kld_unavailable_sparse_support"
		}
	}
	type aggKey struct {
		hostLabel string
		host      string
		model     string
		reqK      string
		reqV      string
		fa        bool
		qjlK      bool
		qjlV      bool
	}
	acc := map[aggKey][]float64{}
	for _, row := range rows {
		if row.KLDivergenceVsBaseline == nil {
			continue
		}
		key := aggKey{hostLabel: row.HostLabel, host: row.Host, model: row.Model, reqK: row.RequestedCacheTypeK, reqV: row.RequestedCacheTypeV, fa: row.FlashAttentionRequested, qjlK: row.QJLKRequested, qjlV: row.QJLVRequested}
		acc[key] = append(acc[key], *row.KLDivergenceVsBaseline)
	}
	for i := range aggregates {
		key := aggKey{hostLabel: aggregates[i].HostLabel, host: aggregates[i].Host, model: aggregates[i].Model, reqK: aggregates[i].RequestedCacheTypeK, reqV: aggregates[i].RequestedCacheTypeV, fa: aggregates[i].FlashAttentionRequested, qjlK: aggregates[i].QJLKRequested, qjlV: aggregates[i].QJLVRequested}
		values := acc[key]
		if len(values) == 0 {
			continue
		}
		sum := 0.0
		for _, value := range values {
			sum += value
		}
		mean := sum / float64(len(values))
		aggregates[i].KLDivergenceVsBaseline = &mean
	}
}

func writeJSONL(path string, rows []chunkResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	return nil
}

func writeCSV(path string, rows []aggregateResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()

	header := []string{
		"host_label", "host", "model", "model_size_label",
		"requested_cache_type_k", "requested_cache_type_v",
		"effective_cache_type_k", "effective_cache_type_v",
		"symmetric_requested", "symmetric_effective",
		"flash_attention_requested", "flash_attention_effective",
		"qjl_k_requested", "qjl_k_effective", "qjl_v_requested", "qjl_v_effective",
		"path_kind", "prompt_tps", "decode_tps", "wall_time_s",
		"kv_buffer_bytes_estimate", "gpu_vram_used_bytes", "gpu_vram_free_bytes", "host_ram_used_bytes",
		"telemetry_path_jsonl", "peak_host_ram_bytes", "peak_gpu_vram_bytes_total", "peak_gpu_vram_bytes_by_gpu", "avg_prompt_tps", "avg_decode_tps", "max_prompt_tps", "max_decode_tps", "eta_confidence", "progress_samples", "live_status_enabled", "telemetry_sampling_interval_sec",
		"fit_status", "corruption_status", "correctness_status", "fallback_reason", "notes", "residency_kind",
		"token_count", "negative_log_likelihood", "perplexity", "ppl_delta_vs_baseline", "kl_divergence_vs_baseline", "chunk_count", "status", "error",
	}
	if err := w.Write(header); err != nil {
		return err
	}
	for _, row := range rows {
		record := []string{
			row.HostLabel,
			row.Host,
			row.Model,
			row.ModelSizeLabel,
			row.RequestedCacheTypeK,
			row.RequestedCacheTypeV,
			row.EffectiveCacheTypeK,
			row.EffectiveCacheTypeV,
			fmt.Sprintf("%t", row.SymmetricRequested),
			fmt.Sprintf("%t", row.SymmetricEffective),
			fmt.Sprintf("%t", row.FlashAttentionRequested),
			fmt.Sprintf("%t", row.FlashAttentionEffective),
			fmt.Sprintf("%t", row.QJLKRequested),
			fmt.Sprintf("%t", row.QJLKEffective),
			fmt.Sprintf("%t", row.QJLVRequested),
			fmt.Sprintf("%t", row.QJLVEffective),
			row.PathKind,
			fmt.Sprintf("%.4f", row.PromptTPS),
			fmt.Sprintf("%.4f", row.DecodeTPS),
			fmt.Sprintf("%.4f", row.WallTimeS),
			formatInt64(row.KVBufferBytesEstimate),
			formatInt64(row.GPUVRAMUsedBytes),
			formatInt64(row.GPUVRAMFreeBytes),
			formatInt64(row.HostRAMUsedBytes),
			row.TelemetryPathJSONL,
			formatInt64(row.PeakHostRAMBytes),
			formatInt64(row.PeakGPUVRAMBytesTotal),
			row.PeakGPUVRAMBytesByGPU,
			formatFloat(row.AvgPromptTPS),
			formatFloat(row.AvgDecodeTPS),
			formatFloat(row.MaxPromptTPS),
			formatFloat(row.MaxDecodeTPS),
			row.ETAConfidence,
			fmt.Sprintf("%d", row.ProgressSamples),
			fmt.Sprintf("%t", row.LiveStatusEnabled),
			fmt.Sprintf("%d", row.TelemetrySamplingSec),
			row.FitStatus,
			row.CorruptionStatus,
			row.CorrectnessStatus,
			row.FallbackReason,
			row.Notes,
			row.ResidencyKind,
			fmt.Sprintf("%d", row.TokenCount),
			fmt.Sprintf("%.6f", row.NegativeLogLikelihood),
			fmt.Sprintf("%.6f", row.Perplexity),
			formatFloat(row.PPLDeltaVsBaseline),
			formatFloat(row.KLDivergenceVsBaseline),
			fmt.Sprintf("%d", row.ChunkCount),
			row.Status,
			row.Error,
		}
		if err := w.Write(record); err != nil {
			return err
		}
	}
	return w.Error()
}

func writeMarkdown(path string, rows []aggregateResult) error {
	var b strings.Builder
	b.WriteString("# PPL Benchmark Summary\n\n")
	b.WriteString("| Host | Model | Requested K/V | Effective K/V | FA req/eff | QJL K req/eff | QJL V req/eff | PPL | Delta vs baseline | KLD vs baseline | pp tok/s | tg tok/s | GPU VRAM | Host RAM | Residency | Correctness | Status |\n")
	b.WriteString("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|\n")
	for _, row := range rows {
		b.WriteString(fmt.Sprintf(
			"| %s | %s | %s/%s | %s/%s | %t/%t | %t/%t | %t/%t | %.4f | %s | %s | %.2f | %.2f | %s | %s | %s | %s | %s |\n",
			row.HostLabel,
			row.Model,
			row.RequestedCacheTypeK,
			row.RequestedCacheTypeV,
			firstNonEmpty(row.EffectiveCacheTypeK, row.RequestedCacheTypeK),
			firstNonEmpty(row.EffectiveCacheTypeV, row.RequestedCacheTypeV),
			row.FlashAttentionRequested,
			row.FlashAttentionEffective,
			row.QJLKRequested,
			row.QJLKEffective,
			row.QJLVRequested,
			row.QJLVEffective,
			row.Perplexity,
			formatFloat(row.PPLDeltaVsBaseline),
			formatFloat(row.KLDivergenceVsBaseline),
			row.PromptTPS,
			row.DecodeTPS,
			formatInt64(row.GPUVRAMUsedBytes),
			formatInt64(row.HostRAMUsedBytes),
			row.ResidencyKind,
			row.CorrectnessStatus,
			row.Status,
		))
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func splitChunkPromptTarget(chunk string) (string, string, bool) {
	fields := strings.Fields(chunk)
	if len(fields) < 4 {
		return "", "", false
	}
	split := max(1, (len(fields)*3)/4)
	if split >= len(fields) {
		split = len(fields) - 1
	}
	prompt := strings.Join(fields[:split], " ")
	target := strings.Join(fields[split:], " ")
	if strings.TrimSpace(prompt) == "" || strings.TrimSpace(target) == "" {
		return "", "", false
	}
	return prompt, target, true
}

func computeSparseKLD(baseline, candidate []api.Logprob) (float64, bool) {
	if len(baseline) == 0 || len(candidate) == 0 || len(baseline) != len(candidate) {
		return 0, false
	}
	total := 0.0
	for i := range baseline {
		pDist := sparseDistribution(candidate[i])
		qDist := sparseDistribution(baseline[i])
		if len(pDist) == 0 || len(qDist) == 0 {
			return 0, false
		}
		kld := 0.0
		for token, p := range pDist {
			if p <= 0 {
				continue
			}
			q, ok := qDist[token]
			if !ok || q <= 0 {
				return 0, false
			}
			kld += p * math.Log(p/q)
		}
		total += kld
	}
	return total / float64(len(baseline)), true
}

func sparseDistribution(lp api.Logprob) map[string]float64 {
	dist := map[string]float64{}
	dist[lp.Token] = math.Exp(lp.Logprob)
	for _, top := range lp.TopLogprobs {
		dist[top.Token] = math.Exp(top.Logprob)
	}
	sum := 0.0
	for _, value := range dist {
		sum += value
	}
	if sum <= 0 {
		return nil
	}
	for token, value := range dist {
		dist[token] = value / sum
	}
	return dist
}

func splitCSV(v string) []string {
	raw := strings.Split(v, ",")
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		item = strings.TrimSpace(item)
		if item != "" {
			out = append(out, item)
		}
	}
	return out
}

func parseSupportModes(v string, hostCount int) ([]hostKVSupportMode, error) {
	if strings.TrimSpace(v) == "" {
		out := make([]hostKVSupportMode, hostCount)
		for i := 0; i < hostCount; i++ {
			if i == 0 {
				out[i] = hostKVSupportLegacy
			} else {
				out[i] = hostKVSupportRequest
			}
		}
		return out, nil
	}
	items := splitCSV(v)
	if len(items) != hostCount {
		return nil, fmt.Errorf("--host-kv-support count must match hosts")
	}
	out := make([]hostKVSupportMode, 0, len(items))
	for _, item := range items {
		switch hostKVSupportMode(item) {
		case hostKVSupportLegacy, hostKVSupportRequest:
			out = append(out, hostKVSupportMode(item))
		default:
			return nil, fmt.Errorf("invalid host kv support mode %q", item)
		}
	}
	return out, nil
}

func parseFAModes(v string) []bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "on":
		return []bool{true}
	case "off", "":
		return []bool{false}
	default:
		return []bool{false, true}
	}
}

func parseQJLModes(v string) []bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "on":
		return []bool{true}
	case "both":
		return []bool{false, true}
	default:
		return []bool{false}
	}
}

func splitKVMode(mode string) (string, string, bool) {
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode == "" {
		return "f16", "f16", true
	}
	if !strings.Contains(mode, "/") {
		return mode, mode, true
	}
	parts := strings.Split(mode, "/")
	if len(parts) != 2 {
		return "", "", false
	}
	return strings.TrimSpace(parts[0]), strings.TrimSpace(parts[1]), true
}

func loadCorpusChunks(path string, chunkTokens, maxChunks int) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	words := strings.Fields(string(data))
	if len(words) == 0 {
		return nil, errors.New("corpus is empty")
	}
	var chunks []string
	for i := 0; i < len(words) && len(chunks) < maxChunks; i += chunkTokens {
		end := min(len(words), i+chunkTokens)
		chunks = append(chunks, strings.Join(words[i:end], " "))
	}
	return chunks, nil
}

func ensureOutputDirs(cfg config) error {
	seen := map[string]struct{}{}
	for _, path := range []string{cfg.OutputDir, cfg.TelemetryDir, filepath.Dir(cfg.OutputPath), filepath.Dir(cfg.JSONLPath), filepath.Dir(cfg.MarkdownPath)} {
		if path == "" {
			continue
		}
		if _, ok := seen[path]; ok {
			continue
		}
		seen[path] = struct{}{}
		if err := os.MkdirAll(path, 0o755); err != nil {
			return err
		}
	}
	return nil
}

func classifyPPLStatus(err error) string {
	var statusErr api.StatusError
	if errors.As(err, &statusErr) {
		if statusErr.StatusCode == 404 || statusErr.StatusCode == 501 || statusErr.StatusCode == 400 {
			return "unsupported"
		}
	}
	return "failed"
}

func sanitizeFileName(v string) string {
	replacer := strings.NewReplacer("/", "_", "\\", "_", ":", "_", " ", "_")
	return replacer.Replace(v)
}

func deriveSizeLabel(model, override string) string {
	if override != "" {
		return override
	}
	lower := strings.ToLower(model)
	switch {
	case strings.Contains(lower, "397b"):
		return "397b"
	case strings.Contains(lower, "32b"):
		return "32b"
	case strings.Contains(lower, "30b"):
		return "30b"
	case strings.Contains(lower, "27b"):
		return "27b"
	case strings.Contains(lower, "9b"):
		return "9b"
	case strings.Contains(lower, "7b"):
		return "7b"
	case strings.Contains(lower, "1.7b"):
		return "1.7b"
	default:
		return "unknown"
	}
}

type pplLiveSample struct {
	Timestamp               time.Time `json:"timestamp"`
	Stage                   string    `json:"stage"`
	HostLabel               string    `json:"host_label"`
	Model                   string    `json:"model"`
	RequestedCacheTypeK     string    `json:"requested_cache_type_k"`
	RequestedCacheTypeV     string    `json:"requested_cache_type_v"`
	EffectiveCacheTypeK     string    `json:"effective_cache_type_k,omitempty"`
	EffectiveCacheTypeV     string    `json:"effective_cache_type_v,omitempty"`
	FlashAttentionRequested bool      `json:"flash_attention_requested"`
	FlashAttentionEffective bool      `json:"flash_attention_effective"`
	QJLKRequested           bool      `json:"qjl_k_requested"`
	QJLKEffective           bool      `json:"qjl_k_effective"`
	QJLVRequested           bool      `json:"qjl_v_requested"`
	QJLVEffective           bool      `json:"qjl_v_effective"`
	ChunkIndex              int       `json:"chunk_index"`
	ChunkTotal              int       `json:"chunk_total"`
	TestIndex               int       `json:"test_index"`
	TestTotal               int       `json:"test_total"`
	ElapsedSec              float64   `json:"elapsed_sec"`
	ETAConfidence           string    `json:"eta_confidence,omitempty"`
	PromptTPS               float64   `json:"prompt_tps,omitempty"`
	DecodeTPS               float64   `json:"decode_tps,omitempty"`
	FallbackReason          string    `json:"fallback_reason,omitempty"`
	Status                  string    `json:"status,omitempty"`
	Error                   string    `json:"error,omitempty"`
}

type pplLiveSummary struct {
	TelemetryPath        string
	AvgPromptTPS         *float64
	AvgDecodeTPS         *float64
	MaxPromptTPS         *float64
	MaxDecodeTPS         *float64
	ETAConfidence        string
	ProgressSamples      int
	LiveStatusEnabled    bool
	TelemetrySamplingSec int
}

type pplLiveSession struct {
	cfg                     config
	path                    string
	writer                  *os.File
	host                    targetHost
	row                     chunkResult
	chunkIndex              int
	chunkTotal              int
	testIndex               int
	testTotal               int
	stage                   string
	startedAt               time.Time
	effectiveCacheTypeK     string
	effectiveCacheTypeV     string
	flashAttentionEffective bool
	qjlKEffective           bool
	qjlVEffective           bool
	fallbackReason          string
	promptTPS               float64
	decodeTPS               float64
	progressSamples         int
	closed                  bool
	stopCh                  chan struct{}
}

func newPPLLiveSession(cfg config, host targetHost, row chunkResult, chunkIndex, chunkTotal, testIndex, testTotal int) *pplLiveSession {
	path := filepath.Join(cfg.TelemetryDir, "benchppl", fmt.Sprintf("%s_%s_chunk%03d.telemetry.jsonl", sanitizeFileName(host.Label), sanitizeFileName(row.RequestedCacheTypeK+"_"+row.RequestedCacheTypeV), chunkIndex))
	var file *os.File
	if cfg.LiveTelemetry {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err == nil {
			if f, err := os.Create(path); err == nil {
				file = f
			}
		}
	}
	s := &pplLiveSession{
		cfg:        cfg,
		path:       path,
		writer:     file,
		host:       host,
		row:        row,
		chunkIndex: chunkIndex,
		chunkTotal: chunkTotal,
		testIndex:  testIndex,
		testTotal:  testTotal,
		stage:      "loading",
		startedAt:  time.Now(),
		stopCh:     make(chan struct{}),
	}
	go s.run()
	return s
}

func (s *pplLiveSession) run() {
	interval := time.Duration(max(1, s.cfg.LiveSampleSec)) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	s.emit("")
	for {
		select {
		case <-ticker.C:
			s.emit("")
		case <-s.stopCh:
			s.emit("")
			return
		}
	}
}

func (s *pplLiveSession) SetStage(stage string) {
	if stage != "" {
		s.stage = stage
	}
}

func (s *pplLiveSession) SetEffective(kEff, vEff string, faEff, qjlKEff, qjlVEff bool, fallback string, promptTPS, decodeTPS float64) {
	s.effectiveCacheTypeK = kEff
	s.effectiveCacheTypeV = vEff
	s.flashAttentionEffective = faEff
	s.qjlKEffective = qjlKEff
	s.qjlVEffective = qjlVEff
	s.fallbackReason = fallback
	s.promptTPS = promptTPS
	s.decodeTPS = decodeTPS
}

func (s *pplLiveSession) emit(status string) {
	s.progressSamples++
	sample := pplLiveSample{
		Timestamp:               time.Now().UTC(),
		Stage:                   s.stage,
		HostLabel:               s.host.Label,
		Model:                   s.row.Model,
		RequestedCacheTypeK:     s.row.RequestedCacheTypeK,
		RequestedCacheTypeV:     s.row.RequestedCacheTypeV,
		EffectiveCacheTypeK:     s.effectiveCacheTypeK,
		EffectiveCacheTypeV:     s.effectiveCacheTypeV,
		FlashAttentionRequested: s.row.FlashAttentionRequested,
		FlashAttentionEffective: s.flashAttentionEffective,
		QJLKRequested:           s.row.QJLKRequested,
		QJLKEffective:           s.qjlKEffective,
		QJLVRequested:           s.row.QJLVRequested,
		QJLVEffective:           s.qjlVEffective,
		ChunkIndex:              s.chunkIndex,
		ChunkTotal:              s.chunkTotal,
		TestIndex:               s.testIndex,
		TestTotal:               s.testTotal,
		ElapsedSec:              time.Since(s.startedAt).Seconds(),
		ETAConfidence:           "low",
		PromptTPS:               s.promptTPS,
		DecodeTPS:               s.decodeTPS,
		FallbackReason:          s.fallbackReason,
		Status:                  status,
	}
	if s.writer != nil {
		enc := json.NewEncoder(s.writer)
		_ = enc.Encode(sample)
		_ = s.writer.Sync()
	}
	if liveStatusEnabled(s.cfg.LiveStatus) {
		fmt.Fprintf(os.Stderr, "[ppl %02d/%02d] host=%s model=%s stage=%s kv=%s/%s eff=%s/%s fa=%t/%t qjl=%t/%t vqjl=%t/%t chunk=%d/%d elapsed=%s eta=%s prompt_tps=%.1f decode_tps=%.1f fallback=%s\n",
			s.testIndex, max(1, s.testTotal), s.host.Label, s.row.Model, s.stage, s.row.RequestedCacheTypeK, s.row.RequestedCacheTypeV,
			firstNonEmpty(s.effectiveCacheTypeK, s.row.RequestedCacheTypeK), firstNonEmpty(s.effectiveCacheTypeV, s.row.RequestedCacheTypeV),
			s.row.FlashAttentionRequested, s.flashAttentionEffective, s.row.QJLKRequested, s.qjlKEffective, s.row.QJLVRequested, s.qjlVEffective,
			s.chunkIndex, max(1, s.chunkTotal), formatElapsed(time.Since(s.startedAt)), "low", s.promptTPS, s.decodeTPS, firstNonEmpty(s.fallbackReason, "none"))
	}
}

func (s *pplLiveSession) Close(status, errText string, row *chunkResult) pplLiveSummary {
	if s.closed {
		return pplLiveSummary{TelemetryPath: s.path, ETAConfidence: "low", ProgressSamples: s.progressSamples, LiveStatusEnabled: liveStatusEnabled(s.cfg.LiveStatus), TelemetrySamplingSec: max(1, s.cfg.LiveSampleSec)}
	}
	s.closed = true
	close(s.stopCh)
	if s.writer != nil {
		_ = s.writer.Close()
	}
	if row != nil {
		row.Error = firstNonEmpty(row.Error, errText)
	}
	var avgPrompt, avgDecode, maxPrompt, maxDecode *float64
	if s.promptTPS > 0 {
		v := s.promptTPS
		avgPrompt, maxPrompt = &v, &v
	}
	if s.decodeTPS > 0 {
		v := s.decodeTPS
		avgDecode, maxDecode = &v, &v
	}
	return pplLiveSummary{
		TelemetryPath:        s.path,
		AvgPromptTPS:         avgPrompt,
		AvgDecodeTPS:         avgDecode,
		MaxPromptTPS:         maxPrompt,
		MaxDecodeTPS:         maxDecode,
		ETAConfidence:        "low",
		ProgressSamples:      s.progressSamples,
		LiveStatusEnabled:    liveStatusEnabled(s.cfg.LiveStatus),
		TelemetrySamplingSec: max(1, s.cfg.LiveSampleSec),
	}
}

func applyPPLLiveSummary(row *chunkResult, summary pplLiveSummary) {
	row.TelemetryPathJSONL = summary.TelemetryPath
	row.AvgPromptTPS = summary.AvgPromptTPS
	row.AvgDecodeTPS = summary.AvgDecodeTPS
	row.MaxPromptTPS = summary.MaxPromptTPS
	row.MaxDecodeTPS = summary.MaxDecodeTPS
	row.ETAConfidence = summary.ETAConfidence
	row.ProgressSamples = summary.ProgressSamples
	row.LiveStatusEnabled = summary.LiveStatusEnabled
	row.TelemetrySamplingSec = summary.TelemetrySamplingSec
}

func countPPLCases(cfg config, chunks []string) int {
	return max(1, len(cfg.Hosts)*len(cfg.KVModes)*len(cfg.FAModes)*len(cfg.QJLKModes)*len(cfg.QJLVModes)*len(chunks))
}

func liveStatusEnabled(mode string) bool {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "off":
		return false
	default:
		return true
	}
}

func formatElapsed(d time.Duration) string {
	total := int(d.Seconds())
	return fmt.Sprintf("%02d:%02d:%02d", total/3600, (total%3600)/60, total%60)
}

func formatInt64(v *int64) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%d", *v)
}

func formatFloat(v *float64) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%.6f", *v)
}

func int64Ptr(v int64) *int64 {
	return &v
}

func maxInt64Ptr(a, b *int64) *int64 {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	if *b > *a {
		return b
	}
	return a
}

func chooseFloatPtr(a, b *float64) *float64 {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}
	if *b > *a {
		return b
	}
	return a
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
