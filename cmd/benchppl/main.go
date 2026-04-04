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
	CorpusPath     string
	ChunkTokens    int
	Chunks         int
	Seed           int
	OutputPath     string
	JSONLPath      string
	MarkdownPath   string
	OutputDir      string
	ModelSizeLabel string
	GatePPLDelta   float64
}

type rawFlags struct {
	hosts          *string
	hostLabels     *string
	hostKVSupport  *string
	model          *string
	kvModes        *string
	faModes        *string
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
	gatePPLDelta   *float64
}

type chunkResult struct {
	HostLabel               string    `json:"host_label"`
	Host                    string    `json:"host"`
	Model                   string    `json:"model"`
	ModelFamily             string    `json:"model_family,omitempty"`
	ModelArch               string    `json:"model_arch,omitempty"`
	ModelSizeLabel          string    `json:"model_size_label,omitempty"`
	RequestedCacheTypeK     string    `json:"requested_cache_type_k"`
	RequestedCacheTypeV     string    `json:"requested_cache_type_v"`
	EffectiveCacheTypeK     string    `json:"effective_cache_type_k,omitempty"`
	EffectiveCacheTypeV     string    `json:"effective_cache_type_v,omitempty"`
	SymmetricRequested      bool      `json:"symmetric_requested"`
	SymmetricEffective      bool      `json:"symmetric_effective"`
	FlashAttentionRequested bool      `json:"flash_attention_requested"`
	FlashAttentionEffective bool      `json:"flash_attention_effective"`
	PathKind                string    `json:"path_kind,omitempty"`
	ContextRequested        int       `json:"context_requested,omitempty"`
	ContextEffective        int       `json:"context_effective,omitempty"`
	PromptTokens            int       `json:"prompt_tokens,omitempty"`
	GeneratedTokens         int       `json:"generated_tokens,omitempty"`
	PromptTPS               float64   `json:"prompt_tps,omitempty"`
	DecodeTPS               float64   `json:"decode_tps,omitempty"`
	WallTimeS               float64   `json:"wall_time_s,omitempty"`
	KVBufferBytesEstimate   *int64    `json:"kv_buffer_bytes_estimate,omitempty"`
	GPUVRAMUsedBytes        *int64    `json:"gpu_vram_used_bytes,omitempty"`
	GPUVRAMFreeBytes        *int64    `json:"gpu_vram_free_bytes,omitempty"`
	HostRAMUsedBytes        *int64    `json:"host_ram_used_bytes,omitempty"`
	FitStatus               string    `json:"fit_status,omitempty"`
	CorruptionStatus        string    `json:"corruption_status,omitempty"`
	CorrectnessStatus       string    `json:"correctness_status,omitempty"`
	FallbackReason          string    `json:"fallback_reason,omitempty"`
	Notes                   string    `json:"notes,omitempty"`
	ResidencyKind           string    `json:"residency_kind,omitempty"`
	ChunkIndex              int       `json:"chunk_index"`
	TokenCount              int       `json:"token_count"`
	NegativeLogLikelihood   float64   `json:"negative_log_likelihood,omitempty"`
	Perplexity              float64   `json:"perplexity,omitempty"`
	PPLDeltaVsBaseline      *float64  `json:"ppl_delta_vs_baseline,omitempty"`
	Status                  string    `json:"status"`
	Error                   string    `json:"error,omitempty"`
	RecordedAt              time.Time `json:"recorded_at"`
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
	Status                  string
	Error                   string
	ChunkCount              int
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

	if err := checkPPLGate(cfg, aggregates); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: PPL gate failed: %v\n", err)
		os.Exit(1)
	}
}

func parseFlags() rawFlags {
	return rawFlags{
		hosts:          flag.String("hosts", "", "Comma-separated Ollama hosts"),
		hostLabels:     flag.String("host-labels", "", "Comma-separated host labels"),
		hostKVSupport:  flag.String("host-kv-support", "", "Comma-separated host KV support modes [legacy|request]"),
		model:          flag.String("model", "", "Model to benchmark"),
		kvModes:        flag.String("kv-modes", "f16,q8_0,q4_0,tq25,tq35,q8_0/tq35", "Comma-separated KV modes"),
		faModes:        flag.String("fa-modes", "both", "Flash Attention modes [on|off|both]"),
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
		gatePPLDelta:   flag.Float64("gate-ppl-delta", 0, "Exit non-zero if any TQ mode PPL delta vs f16 baseline exceeds this threshold (0 = disabled)"),
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
		CorpusPath:     strings.TrimSpace(*r.corpus),
		ChunkTokens:    max(1, *r.chunkTokens),
		Chunks:         max(1, *r.chunks),
		Seed:           *r.seed,
		OutputDir:      strings.TrimSpace(*r.outputDir),
		ModelSizeLabel: strings.TrimSpace(*r.modelSizeLabel),
		GatePPLDelta:   *r.gatePPLDelta,
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

	return cfg, nil
}

func runPPLBench(cfg config, chunks []string) ([]chunkResult, []aggregateResult) {
	var rows []chunkResult
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
					rows = append(rows, scoreChunk(cfg, host, kType, vType, faRequested, idx, chunk))
				}
			}
		}
	}

	aggregates := aggregateChunkResults(rows)
	applyBaselinePPLDeltas(aggregates)
	return rows, aggregates
}

func scoreChunk(cfg config, host targetHost, kType string, vType string, faRequested bool, idx int, chunk string) chunkResult {
	row := chunkResult{
		HostLabel:               host.Label,
		Host:                    host.BaseURL,
		Model:                   cfg.Model,
		ModelSizeLabel:          deriveSizeLabel(cfg.Model, cfg.ModelSizeLabel),
		RequestedCacheTypeK:     kType,
		RequestedCacheTypeV:     vType,
		SymmetricRequested:      strings.EqualFold(kType, vType),
		FlashAttentionRequested: faRequested,
		FitStatus:               string(tqbenchschema.FitStatusFailed),
		CorruptionStatus:        string(tqbenchschema.CorruptionStatusPass),
		CorrectnessStatus:       string(tqbenchschema.CorrectnessStatusFail),
		ResidencyKind:           string(tqbenchschema.ResidencyUnknown),
		ChunkIndex:              idx,
		RecordedAt:              time.Now().UTC(),
		Notes:                   "chunk scoring uses raw prompt+target bench route",
	}

	options := map[string]any{
		"num_ctx":         max(4096, cfg.ChunkTokens*4),
		"cache_type_k":    kType,
		"cache_type_v":    vType,
		"seed":            cfg.Seed,
		"temperature":     0,
		"top_p":           1,
		"repeat_penalty":  1,
		"num_predict":     max(1, len(strings.Fields(chunk))),
		"flash_attention": faRequested,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	start := time.Now()
	resp, err := host.Client.BenchScore(ctx, &api.BenchScoreRequest{
		Model:   cfg.Model,
		Prompt:  "",
		Target:  chunk,
		Options: options,
	})
	row.WallTimeS = time.Since(start).Seconds()
	if err != nil {
		row.Status = classifyPPLStatus(err)
		row.Error = err.Error()
		if row.Status == "unsupported" {
			row.FitStatus = string(tqbenchschema.FitStatusUnsupported)
			row.CorrectnessStatus = string(tqbenchschema.CorrectnessStatusUnsupported)
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
	}
	aggMap := map[key]*aggregateResult{}
	for _, row := range rows {
		k := key{hostLabel: row.HostLabel, host: row.Host, model: row.Model, reqK: row.RequestedCacheTypeK, reqV: row.RequestedCacheTypeV, fa: row.FlashAttentionRequested}
		if aggMap[k] == nil {
			aggMap[k] = &aggregateResult{
				HostLabel:               row.HostLabel,
				Host:                    row.Host,
				Model:                   row.Model,
				ModelSizeLabel:          row.ModelSizeLabel,
				RequestedCacheTypeK:     row.RequestedCacheTypeK,
				RequestedCacheTypeV:     row.RequestedCacheTypeV,
				FlashAttentionRequested: row.FlashAttentionRequested,
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
		cur.PathKind = firstNonEmpty(cur.PathKind, row.PathKind)
		cur.FallbackReason = firstNonEmpty(cur.FallbackReason, row.FallbackReason)
		cur.Notes = firstNonEmpty(cur.Notes, row.Notes)
		cur.KVBufferBytesEstimate = maxInt64Ptr(cur.KVBufferBytesEstimate, row.KVBufferBytesEstimate)
		cur.GPUVRAMUsedBytes = maxInt64Ptr(cur.GPUVRAMUsedBytes, row.GPUVRAMUsedBytes)
		cur.GPUVRAMFreeBytes = maxInt64Ptr(cur.GPUVRAMFreeBytes, row.GPUVRAMFreeBytes)
		cur.HostRAMUsedBytes = maxInt64Ptr(cur.HostRAMUsedBytes, row.HostRAMUsedBytes)
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
	}

	out := make([]aggregateResult, 0, len(aggMap))
	for _, agg := range aggMap {
		if agg.ChunkCount > 0 {
			agg.PromptTPS /= float64(agg.ChunkCount)
			agg.DecodeTPS /= float64(agg.ChunkCount)
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

// checkPPLGate returns an error if any TQ-mode row's PPL delta vs the f16 baseline
// exceeds cfg.GatePPLDelta. A delta of 0 disables the gate. This enforces the
// paper's quality-neutral claim: tq35 should add no more than ~0.5 PPL points
// vs full-precision on standard corpora.
func checkPPLGate(cfg config, rows []aggregateResult) error {
	if cfg.GatePPLDelta <= 0 {
		return nil
	}
	var failures []string
	for _, row := range rows {
		isTQ := strings.HasPrefix(row.RequestedCacheTypeK, "tq") || strings.HasPrefix(row.RequestedCacheTypeV, "tq")
		if !isTQ || row.PPLDeltaVsBaseline == nil {
			continue
		}
		if *row.PPLDeltaVsBaseline > cfg.GatePPLDelta {
			failures = append(failures, fmt.Sprintf(
				"host=%s model=%s kv=%s/%s fa=%t: delta=%.4f > threshold=%.4f",
				row.HostLabel, row.Model,
				row.RequestedCacheTypeK, row.RequestedCacheTypeV,
				row.FlashAttentionRequested,
				*row.PPLDeltaVsBaseline, cfg.GatePPLDelta,
			))
		}
	}
	if len(failures) > 0 {
		return fmt.Errorf("PPL delta exceeds --gate-ppl-delta=%.4f for %d row(s):\n  %s",
			cfg.GatePPLDelta, len(failures), strings.Join(failures, "\n  "))
	}
	return nil
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
		"path_kind", "prompt_tps", "decode_tps", "wall_time_s",
		"kv_buffer_bytes_estimate", "gpu_vram_used_bytes", "gpu_vram_free_bytes", "host_ram_used_bytes",
		"fit_status", "corruption_status", "correctness_status", "fallback_reason", "notes", "residency_kind",
		"token_count", "negative_log_likelihood", "perplexity", "ppl_delta_vs_baseline", "chunk_count", "status", "error",
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
			row.PathKind,
			fmt.Sprintf("%.4f", row.PromptTPS),
			fmt.Sprintf("%.4f", row.DecodeTPS),
			fmt.Sprintf("%.4f", row.WallTimeS),
			formatInt64(row.KVBufferBytesEstimate),
			formatInt64(row.GPUVRAMUsedBytes),
			formatInt64(row.GPUVRAMFreeBytes),
			formatInt64(row.HostRAMUsedBytes),
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
	b.WriteString("| Host | Model | Requested K/V | Effective K/V | FA req/eff | PPL | Delta vs baseline | pp tok/s | tg tok/s | GPU VRAM | Host RAM | Residency | Correctness | Status |\n")
	b.WriteString("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|\n")
	for _, row := range rows {
		b.WriteString(fmt.Sprintf(
			"| %s | %s | %s/%s | %s/%s | %t/%t | %.4f | %s | %.2f | %.2f | %s | %s | %s | %s | %s |\n",
			row.HostLabel,
			row.Model,
			row.RequestedCacheTypeK,
			row.RequestedCacheTypeV,
			firstNonEmpty(row.EffectiveCacheTypeK, row.RequestedCacheTypeK),
			firstNonEmpty(row.EffectiveCacheTypeV, row.RequestedCacheTypeV),
			row.FlashAttentionRequested,
			row.FlashAttentionEffective,
			row.Perplexity,
			formatFloat(row.PPLDeltaVsBaseline),
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
	for _, path := range []string{cfg.OutputDir, filepath.Dir(cfg.OutputPath), filepath.Dir(cfg.JSONLPath), filepath.Dir(cfg.MarkdownPath)} {
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

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
