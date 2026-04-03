package main

import (
	"errors"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

type rawFlags struct {
	hosts               *string
	hostLabels          *string
	hostKVSupport       *string
	model               *string
	kvModes             *string
	profile             *string
	workloads           *string
	numCtx              *string
	concurrency         *string
	promptTokens        *int
	maxTokens           *int
	warmup              *int
	epochs              *int
	keepAlive           *float64
	seed                *int
	temperature         *float64
	timeout             *int
	stream              *bool
	failFast            *bool
	probeIntervalMS     *int
	output              *string
	jsonlOutput         *string
	summaryOutput       *string
	markdownOutput      *string
	outputDir           *string
	baselineHost        *string
	turboHost           *string
	contextLadder       *string
	stretchContext      *int
	minFitDecode        *int
	promptFile          *string
	longJSONBytes       *int
	turboMode           *string
	faModes             *string
	repeats             *int
	recallDistances     *string
	toolSuite           *string
	modelFamilyOverride *string
	modelSizeLabel      *string
	targetNativeCtx     *int
	targetYarnCtx       *int
	captureOllamaPS     *bool
	captureGPU          *bool
	captureRunnerRSS    *bool
	debug               *bool
	progress            *string
	progressWidth       *int
}

func parseFlags() rawFlags {
	return rawFlags{
		hosts:               flag.String("hosts", "", "Comma-separated Ollama hosts to benchmark"),
		hostLabels:          flag.String("host-labels", "", "Optional comma-separated labels matching --hosts"),
		hostKVSupport:       flag.String("host-kv-support", "", "Optional comma-separated host KV support modes matching --hosts [legacy|request]"),
		model:               flag.String("model", "", "Model to benchmark"),
		kvModes:             flag.String("kv-modes", "", "Comma-separated KV modes"),
		profile:             flag.String("profile", "quick", "Benchmark profile [quick|full|staircase|impact|regression|turbo-benefit|capacity|spill|large-context|test-matrix|agentic|memory]"),
		workloads:           flag.String("workloads", "", "Comma-separated workloads override"),
		numCtx:              flag.String("num-ctx", "", "Comma-separated context lengths override"),
		concurrency:         flag.String("concurrency", "", "Comma-separated concurrency override"),
		promptTokens:        flag.Int("prompt-tokens", 0, "Override prompt token target for all workloads"),
		maxTokens:           flag.Int("max-tokens", 0, "Override max tokens for all workloads"),
		warmup:              flag.Int("warmup", -1, "Warmup epochs override"),
		epochs:              flag.Int("epochs", -1, "Timed epochs override"),
		keepAlive:           flag.Float64("keep-alive", -1, "Keep alive duration in seconds (-1 keeps model loaded)"),
		seed:                flag.Int("seed", 42, "Seed"),
		temperature:         flag.Float64("temperature", 0, "Temperature"),
		timeout:             flag.Int("timeout", 1800, "Timeout in seconds (0 disables timeout)"),
		stream:              flag.Bool("stream", true, "Use streaming responses"),
		failFast:            flag.Bool("fail-fast", false, "Stop entire sweep on first failure"),
		probeIntervalMS:     flag.Int("probe-interval-ms", 100, "GPU probe interval in milliseconds"),
		output:              flag.String("output", "", "CSV summary output path"),
		jsonlOutput:         flag.String("jsonl-output", "", "JSONL output path"),
		summaryOutput:       flag.String("summary-output", "", "Text summary output path"),
		markdownOutput:      flag.String("markdown-output", "", "Markdown summary output path (defaults to summary output)"),
		outputDir:           flag.String("output-dir", "/results", "Output directory for default outputs"),
		baselineHost:        flag.String("baseline-host", "http://127.0.0.1:11438", "Convenience baseline host URL"),
		turboHost:           flag.String("turbo-host", "http://127.0.0.1:11439", "Convenience TurboQuant host URL"),
		contextLadder:       flag.String("context-ladder", "", "Comma-separated large-context ladder override"),
		stretchContext:      flag.Int("stretch-context", 1000000, "Top requested large-context target for reporting"),
		minFitDecode:        flag.Int("min-fit-decode-tokens", 32, "Minimum decode tokens required for a ladder rung to count as fit"),
		promptFile:          flag.String("prompt-file", "", "Optional file-backed prompt fixture for prompt-file-regression"),
		longJSONBytes:       flag.Int("long-json-bytes", 16384, "Approximate long JSON payload size for retention validation"),
		turboMode:           flag.String("turbo-mode", "tq35", "TurboQuant mode to use for large-context mixed/symmetric lanes [tq25|tq35]"),
		faModes:             flag.String("fa-modes", "", "Flash Attention request modes [on|off|both]"),
		repeats:             flag.Int("repeats", 1, "Repeated deterministic validation runs for corruption-sensitive workloads"),
		recallDistances:     flag.String("recall-distances", "0,200,500,1000,4000,16000", "Comma-separated recall distances"),
		toolSuite:           flag.String("tool-suite", "default", "Agentic/structured-output fixture subset"),
		modelFamilyOverride: flag.String("model-family-override", "", "Optional model family label override for summaries"),
		modelSizeLabel:      flag.String("model-size-label", "", "Optional model size label override for summaries"),
		targetNativeCtx:     flag.Int("target-native-context", 0, "Optional advertised native context length for reporting"),
		targetYarnCtx:       flag.Int("target-yarn-context", 0, "Optional advertised YaRN stretch context length for reporting"),
		captureOllamaPS:     flag.Bool("capture-ollama-ps", true, "Capture `ollama ps` before and after runs when available"),
		captureGPU:          flag.Bool("capture-gpu", true, "Capture GPU metrics via NVML or nvidia-smi"),
		captureRunnerRSS:    flag.Bool("capture-runner-rss", true, "Capture runner RSS when possible"),
		debug:               flag.Bool("debug", false, "Enable debug logging"),
		progress:            flag.String("progress", "auto", "Progress display mode [auto|on|off]"),
		progressWidth:       flag.Int("progress-width", 30, "Progress bar width"),
	}
}

func loadConfig(r rawFlags) (config, error) {
	turboModeValue := "tq35"
	if r.turboMode != nil {
		turboModeValue = *r.turboMode
	}
	contextLadderValue := ""
	if r.contextLadder != nil {
		contextLadderValue = *r.contextLadder
	}
	promptFileValue := ""
	if r.promptFile != nil {
		promptFileValue = *r.promptFile
	}
	stretchContextValue := 1000000
	if r.stretchContext != nil {
		stretchContextValue = *r.stretchContext
	}
	minFitDecodeValue := 32
	if r.minFitDecode != nil {
		minFitDecodeValue = *r.minFitDecode
	}
	longJSONBytesValue := 16384
	if r.longJSONBytes != nil {
		longJSONBytesValue = *r.longJSONBytes
	}
	targetNativeCtxValue := 0
	if r.targetNativeCtx != nil {
		targetNativeCtxValue = *r.targetNativeCtx
	}
	targetYarnCtxValue := 0
	if r.targetYarnCtx != nil {
		targetYarnCtxValue = *r.targetYarnCtx
	}
	repeatsValue := 1
	if r.repeats != nil && *r.repeats > 0 {
		repeatsValue = *r.repeats
	}
	faModesValue := ""
	if r.faModes != nil {
		faModesValue = *r.faModes
	}
	recallDistancesValue := ""
	if r.recallDistances != nil {
		recallDistancesValue = *r.recallDistances
	}
	toolSuiteValue := ""
	if r.toolSuite != nil {
		toolSuiteValue = *r.toolSuite
	}
	baselineHostValue := "http://127.0.0.1:11438"
	if r.baselineHost != nil {
		baselineHostValue = *r.baselineHost
	}
	turboHostValue := "http://127.0.0.1:11439"
	if r.turboHost != nil {
		turboHostValue = *r.turboHost
	}
	modelFamilyOverrideValue := ""
	if r.modelFamilyOverride != nil {
		modelFamilyOverrideValue = *r.modelFamilyOverride
	}
	modelSizeLabelValue := ""
	if r.modelSizeLabel != nil {
		modelSizeLabelValue = *r.modelSizeLabel
	}
	markdownOutputValue := ""
	if r.markdownOutput != nil {
		markdownOutputValue = *r.markdownOutput
	}

	hostsValue := strings.TrimSpace(*r.hosts)
	if hostsValue == "" {
		baseline := strings.TrimSpace(baselineHostValue)
		turbo := strings.TrimSpace(turboHostValue)
		switch {
		case baseline != "" && turbo != "":
			hostsValue = baseline + "," + turbo
		case baseline != "":
			hostsValue = baseline
		case turbo != "":
			hostsValue = turbo
		}
	}
	if strings.TrimSpace(hostsValue) == "" {
		return config{}, errors.New("--hosts is required")
	}
	if strings.TrimSpace(*r.model) == "" {
		return config{}, errors.New("--model is required")
	}

	profile := strings.ToLower(strings.TrimSpace(*r.profile))
	if !slices.Contains([]string{"quick", "full", "staircase", "impact", "regression", "turbo-benefit", "capacity", "spill", "large-context", "test-matrix", "agentic", "memory"}, profile) {
		return config{}, fmt.Errorf("invalid profile %q", *r.profile)
	}

	hostURLs, err := parseCSV(hostsValue)
	if err != nil {
		return config{}, fmt.Errorf("invalid --hosts: %w", err)
	}
	if len(hostURLs) == 0 {
		return config{}, errors.New("--hosts must contain at least one host")
	}

	labels, err := parseOptionalCSV(*r.hostLabels)
	if err != nil {
		return config{}, fmt.Errorf("invalid --host-labels: %w", err)
	}
	if len(labels) > 0 && len(labels) != len(hostURLs) {
		return config{}, errors.New("--host-labels count must match --hosts")
	}
	if len(labels) == 0 {
		labels = defaultHostLabels(len(hostURLs))
	}

	kvSupportModes, err := parseHostKVSupportModes(*r.hostKVSupport, len(hostURLs))
	if err != nil {
		return config{}, fmt.Errorf("invalid --host-kv-support: %w", err)
	}

	clientTimeout := timeoutDuration(*r.timeout)
	hosts := make([]hostTarget, 0, len(hostURLs))
	for i, host := range hostURLs {
		parsed, err := url.Parse(host)
		if err != nil {
			return config{}, fmt.Errorf("invalid host %q: %w", host, err)
		}
		if parsed.Scheme == "" || parsed.Host == "" {
			return config{}, fmt.Errorf("host %q must include scheme and host", host)
		}
		hosts = append(hosts, hostTarget{
			BaseURL:       host,
			Label:         labels[i],
			URL:           parsed,
			Client:        api.NewClient(parsed, &http.Client{Timeout: clientTimeout}),
			KVSupportMode: kvSupportModes[i],
		})
	}

	turboMode, err := normalizeKVMode(turboModeValue)
	if err != nil {
		return config{}, fmt.Errorf("invalid --turbo-mode: %w", err)
	}

	kvModeValue := *r.kvModes
	if strings.TrimSpace(kvModeValue) == "" {
		kvModeValue = strings.Join(defaultKVModes(profile, turboMode), ",")
	}
	kvModes, err := parseOptionalCSV(kvModeValue)
	if err != nil {
		return config{}, fmt.Errorf("invalid --kv-modes: %w", err)
	}
	if len(kvModes) == 0 {
		return config{}, errors.New("--kv-modes must contain at least one mode")
	}
	for i := range kvModes {
		kvModes[i], err = normalizeBenchmarkKVMode(kvModes[i])
		if err != nil {
			return config{}, err
		}
	}

	numCtx := profileContexts(profile)
	if strings.TrimSpace(*r.numCtx) != "" {
		numCtx, err = parseIntCSV(*r.numCtx)
		if err != nil {
			return config{}, fmt.Errorf("invalid --num-ctx: %w", err)
		}
	}

	concurrency := profileConcurrency(profile)
	if strings.TrimSpace(*r.concurrency) != "" {
		concurrency, err = parseIntCSV(*r.concurrency)
		if err != nil {
			return config{}, fmt.Errorf("invalid --concurrency: %w", err)
		}
	}

	workloads := profileWorkloads(profile)
	if strings.TrimSpace(*r.workloads) != "" {
		workloads, err = parseWorkloads(*r.workloads)
		if err != nil {
			return config{}, fmt.Errorf("invalid --workloads: %w", err)
		}
	}

	contextLadder := defaultContextLadder()
	if strings.TrimSpace(contextLadderValue) != "" {
		contextLadder, err = parseIntCSV(contextLadderValue)
		if err != nil {
			return config{}, fmt.Errorf("invalid --context-ladder: %w", err)
		}
	}
	faModes, err := parseFAModes(faModesValue, profile)
	if err != nil {
		return config{}, fmt.Errorf("invalid --fa-modes: %w", err)
	}
	recallDistances, err := parseIntCSVAllowEmpty(recallDistancesValue)
	if err != nil {
		return config{}, fmt.Errorf("invalid --recall-distances: %w", err)
	}

	warmup := profileWarmup(profile)
	if *r.warmup >= 0 {
		warmup = *r.warmup
	}
	epochs := profileEpochs(profile)
	if *r.epochs >= 0 {
		epochs = *r.epochs
	}

	cfg := config{
		Hosts:               hosts,
		Model:               *r.model,
		KVModes:             kvModes,
		TurboMode:           turboMode,
		FAModes:             faModes,
		Repeats:             repeatsValue,
		RecallDistances:     recallDistances,
		ToolSuite:           strings.TrimSpace(toolSuiteValue),
		Profile:             profile,
		Workloads:           workloads,
		NumCtx:              numCtx,
		Concurrency:         concurrency,
		PromptTokens:        *r.promptTokens,
		MaxTokens:           *r.maxTokens,
		Warmup:              warmup,
		Epochs:              epochs,
		KeepAlive:           time.Duration(*r.keepAlive * float64(time.Second)),
		Seed:                *r.seed,
		Temperature:         *r.temperature,
		Timeout:             clientTimeout,
		Stream:              *r.stream,
		FailFast:            *r.failFast,
		ProbeInterval:       time.Duration(*r.probeIntervalMS) * time.Millisecond,
		CaptureOllamaPS:     *r.captureOllamaPS,
		CaptureGPU:          *r.captureGPU,
		CaptureRunnerRSS:    *r.captureRunnerRSS,
		OutputDir:           *r.outputDir,
		BaselineHost:        strings.TrimSpace(baselineHostValue),
		TurboHost:           strings.TrimSpace(turboHostValue),
		ModelFamilyOverride: strings.TrimSpace(modelFamilyOverrideValue),
		ModelSizeLabel:      strings.TrimSpace(modelSizeLabelValue),
		ContextLadder:       contextLadder,
		StretchContext:      stretchContextValue,
		MinFitDecode:        minFitDecodeValue,
		PromptFile:          strings.TrimSpace(promptFileValue),
		LongJSONBytes:       longJSONBytesValue,
		TargetNativeCtx:     targetNativeCtxValue,
		TargetYarnCtx:       targetYarnCtxValue,
		Debug:               *r.debug,
		ProgressWidth:       *r.progressWidth,
	}

	mode, err := parseProgressMode(*r.progress)
	if err != nil {
		return config{}, err
	}
	cfg.ProgressMode = mode

	cfg.OutputPath = *r.output
	cfg.JSONLPath = *r.jsonlOutput
	cfg.SummaryPath = *r.summaryOutput
	cfg.MarkdownPath = markdownOutputValue
	fillDefaultOutputs(&cfg)
	return cfg, nil
}

func timeoutDuration(seconds int) time.Duration {
	if seconds <= 0 {
		return 0
	}
	return time.Duration(seconds) * time.Second
}

func parseProgressMode(value string) (progressMode, error) {
	switch progressMode(strings.ToLower(strings.TrimSpace(value))) {
	case progressAuto, progressOn, progressOff:
		return progressMode(strings.ToLower(strings.TrimSpace(value))), nil
	default:
		return "", fmt.Errorf("invalid progress mode %q", value)
	}
}

func parseHostKVSupportModes(value string, hostCount int) ([]hostKVSupportMode, error) {
	if strings.TrimSpace(value) == "" {
		return defaultHostKVSupportModes(hostCount), nil
	}

	parts, err := parseCSV(value)
	if err != nil {
		return nil, err
	}
	if len(parts) != hostCount {
		return nil, fmt.Errorf("count must match --hosts")
	}

	modes := make([]hostKVSupportMode, 0, len(parts))
	for _, part := range parts {
		switch hostKVSupportMode(strings.ToLower(strings.TrimSpace(part))) {
		case hostKVSupportLegacy:
			modes = append(modes, hostKVSupportLegacy)
		case hostKVSupportRequest:
			modes = append(modes, hostKVSupportRequest)
		default:
			return nil, fmt.Errorf("unknown host kv support mode %q", part)
		}
	}
	return modes, nil
}

func normalizeKVMode(value string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "off", "f16":
		return "f16", nil
	case "q8_0", "q4_0", "tq25", "tq35":
		return strings.ToLower(strings.TrimSpace(value)), nil
	default:
		return "", fmt.Errorf("invalid kv mode %q", value)
	}
}

func normalizeBenchmarkKVMode(value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if strings.Contains(value, "/") {
		kType, vType, ok := splitBenchmarkKVMode(value)
		if !ok {
			return "", fmt.Errorf("invalid kv mode %q", value)
		}
		kType, err := normalizeKVMode(kType)
		if err != nil {
			return "", err
		}
		vType, err = normalizeKVMode(vType)
		if err != nil {
			return "", err
		}
		return kType + "/" + vType, nil
	}
	return normalizeKVMode(value)
}

func parseCSV(value string) ([]string, error) {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, errors.New("empty list item")
		}
		out = append(out, part)
	}
	return out, nil
}

func parseOptionalCSV(value string) ([]string, error) {
	if strings.TrimSpace(value) == "" {
		return nil, nil
	}
	return parseCSV(value)
}

func parseIntCSV(value string) ([]int, error) {
	parts, err := parseCSV(value)
	if err != nil {
		return nil, err
	}
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, err
		}
		if n <= 0 {
			return nil, fmt.Errorf("value %d must be > 0", n)
		}
		out = append(out, n)
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

func parseIntCSVAllowEmpty(value string) ([]int, error) {
	if strings.TrimSpace(value) == "" {
		return nil, nil
	}
	parts, err := parseCSV(value)
	if err != nil {
		return nil, err
	}
	out := make([]int, 0, len(parts))
	for _, part := range parts {
		n, err := strconv.Atoi(part)
		if err != nil {
			return nil, err
		}
		if n < 0 {
			return nil, fmt.Errorf("value %d must be >= 0", n)
		}
		out = append(out, n)
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}

func parseFAModes(value, profile string) ([]bool, error) {
	mode := strings.ToLower(strings.TrimSpace(value))
	if mode == "" {
		switch profile {
		case "large-context", "test-matrix", "agentic", "memory":
			mode = "both"
		default:
			mode = "off"
		}
	}
	switch mode {
	case "on":
		return []bool{true}, nil
	case "off":
		return []bool{false}, nil
	case "both":
		return []bool{false, true}, nil
	default:
		return nil, fmt.Errorf("unknown fa mode %q", value)
	}
}

func parseWorkloads(value string) ([]workloadName, error) {
	parts, err := parseCSV(value)
	if err != nil {
		return nil, err
	}
	out := make([]workloadName, 0, len(parts))
	for _, part := range parts {
		w := workloadName(strings.ToLower(strings.TrimSpace(part)))
		switch w {
		case workloadPrefillHeavy, workloadDecodeGrowth, workloadParallelAmplifier, workloadNearOOMStaircase, workloadFitCeiling, workloadLongContextRecall, workloadLongJSONRetention, workloadPromptFileRegress, workloadDecodeCorruption, workloadAgenticStructured:
			out = append(out, w)
		default:
			return nil, fmt.Errorf("unknown workload %q", part)
		}
	}
	return slices.Compact(out), nil
}

func defaultHostLabels(n int) []string {
	labels := make([]string, n)
	for i := 0; i < n; i++ {
		switch i {
		case 0:
			labels[i] = "baseline"
		case 1:
			labels[i] = "turbo"
		default:
			labels[i] = fmt.Sprintf("host%d", i+1)
		}
	}
	return labels
}

func defaultHostKVSupportModes(n int) []hostKVSupportMode {
	modes := make([]hostKVSupportMode, n)
	for i := 0; i < n; i++ {
		if i == 0 {
			modes[i] = hostKVSupportLegacy
			continue
		}
		modes[i] = hostKVSupportRequest
	}
	return modes
}

func fillDefaultOutputs(cfg *config) {
	outputDir := strings.TrimSpace(cfg.OutputDir)
	if outputDir == "" {
		outputDir = "/results"
		cfg.OutputDir = outputDir
	}
	if cfg.OutputPath == "" {
		cfg.OutputPath = filepath.Join(outputDir, "kvstress_summary.csv")
	}
	if cfg.JSONLPath == "" {
		cfg.JSONLPath = filepath.Join(outputDir, "kvstress_epochs.jsonl")
	}
	if cfg.SummaryPath == "" {
		cfg.SummaryPath = filepath.Join(outputDir, "kvstress_summary.md")
	}
	if cfg.MarkdownPath == "" {
		cfg.MarkdownPath = cfg.SummaryPath
	}
}

func profileContexts(profile string) []int {
	switch profile {
	case "regression":
		return []int{8192}
	case "turbo-benefit":
		return []int{8192, 16384, 32768, 65536}
	case "capacity", "spill":
		return []int{8192, 16384, 32768, 65536}
	case "large-context":
		return defaultContextLadder()
	case "memory":
		return defaultContextLadder()
	case "test-matrix", "agentic":
		return []int{8192, 16384, 32768}
	case "quick":
		return []int{8192, 16384}
	case "impact":
		return []int{16384, 8192}
	case "full":
		return []int{8192, 16384, 32768, 65536}
	case "staircase":
		return []int{8192, 16384, 32768, 65536, 131072, 262144}
	default:
		return []int{8192, 16384}
	}
}

func profileConcurrency(profile string) []int {
	switch profile {
	case "regression":
		return []int{1}
	case "turbo-benefit", "capacity":
		return []int{1, 2, 4}
	case "spill":
		return []int{1, 2, 4, 8}
	case "quick":
		return []int{1, 2}
	case "impact":
		return []int{2, 1}
	case "large-context":
		return []int{1}
	case "test-matrix", "agentic", "memory":
		return []int{1}
	case "full", "staircase":
		return []int{1, 2, 4}
	default:
		return []int{1, 2}
	}
}

func profileWorkloads(profile string) []workloadName {
	switch profile {
	case "regression":
		return []workloadName{workloadDecodeGrowth}
	case "turbo-benefit":
		return []workloadName{workloadPrefillHeavy, workloadDecodeGrowth}
	case "staircase":
		return []workloadName{workloadNearOOMStaircase}
	case "capacity", "spill":
		return []workloadName{workloadNearOOMStaircase}
	case "large-context":
		return []workloadName{workloadFitCeiling, workloadPrefillHeavy, workloadLongContextRecall, workloadDecodeCorruption, workloadPromptFileRegress, workloadLongJSONRetention}
	case "test-matrix":
		return []workloadName{workloadLongContextRecall, workloadDecodeCorruption, workloadPromptFileRegress}
	case "agentic":
		return []workloadName{workloadAgenticStructured}
	case "memory":
		return []workloadName{workloadFitCeiling, workloadPrefillHeavy, workloadLongContextRecall, workloadDecodeCorruption}
	case "impact":
		return []workloadName{workloadPrefillHeavy, workloadDecodeGrowth}
	default:
		return []workloadName{workloadPrefillHeavy, workloadDecodeGrowth, workloadParallelAmplifier}
	}
}

func profileWarmup(profile string) int {
	return 1
}

func profileEpochs(profile string) int {
	switch profile {
	case "regression":
		return 8
	case "turbo-benefit":
		return 3
	case "capacity", "spill":
		return 2
	case "quick":
		return 3
	case "impact":
		return 1
	case "staircase":
		return 2
	case "large-context":
		return 1
	case "test-matrix", "agentic", "memory":
		return 1
	default:
		return 6
	}
}

func defaultKVModes(profile string, turboMode string) []string {
	switch profile {
	case "regression":
		return []string{"f16"}
	case "turbo-benefit", "capacity", "spill":
		return []string{"f16", "tq25", "tq35"}
	case "large-context":
		return []string{"f16", "q8_0", "q8_0/" + turboMode, turboMode}
	case "test-matrix", "agentic", "memory":
		return []string{"f16", "q8_0", "q4_0", "tq25", "tq35", "q8_0/" + turboMode}
	default:
		return []string{"f16", "q8_0", "q4_0", "tq25", "tq35"}
	}
}

func defaultContextLadder() []int {
	return []int{1000000, 750000, 500000, 262144, 128000, 104000, 96000, 65536, 32768, 16384}
}

func ensureOutputDirs(cfg config) error {
	dirs := []string{
		filepath.Dir(cfg.OutputPath),
		filepath.Dir(cfg.JSONLPath),
		filepath.Dir(cfg.SummaryPath),
		filepath.Dir(cfg.MarkdownPath),
	}
	for _, dir := range dirs {
		if dir == "." || dir == "" {
			continue
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return nil
}
