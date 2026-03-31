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
	hosts            *string
	hostLabels       *string
	hostKVSupport    *string
	model            *string
	kvModes          *string
	profile          *string
	workloads        *string
	numCtx           *string
	concurrency      *string
	promptTokens     *int
	maxTokens        *int
	warmup           *int
	epochs           *int
	keepAlive        *float64
	seed             *int
	temperature      *float64
	timeout          *int
	stream           *bool
	failFast         *bool
	probeIntervalMS  *int
	output           *string
	jsonlOutput      *string
	summaryOutput    *string
	outputDir        *string
	captureOllamaPS  *bool
	captureGPU       *bool
	captureRunnerRSS *bool
	debug            *bool
	progress         *string
	progressWidth    *int
}

func parseFlags() rawFlags {
	return rawFlags{
		hosts:            flag.String("hosts", "", "Comma-separated Ollama hosts to benchmark"),
		hostLabels:       flag.String("host-labels", "", "Optional comma-separated labels matching --hosts"),
		hostKVSupport:    flag.String("host-kv-support", "", "Optional comma-separated host KV support modes matching --hosts [legacy|request]"),
		model:            flag.String("model", "", "Model to benchmark"),
		kvModes:          flag.String("kv-modes", "", "Comma-separated KV modes"),
		profile:          flag.String("profile", "quick", "Benchmark profile [quick|full|staircase|impact|regression|turbo-benefit|capacity|spill]"),
		workloads:        flag.String("workloads", "", "Comma-separated workloads override"),
		numCtx:           flag.String("num-ctx", "", "Comma-separated context lengths override"),
		concurrency:      flag.String("concurrency", "", "Comma-separated concurrency override"),
		promptTokens:     flag.Int("prompt-tokens", 0, "Override prompt token target for all workloads"),
		maxTokens:        flag.Int("max-tokens", 0, "Override max tokens for all workloads"),
		warmup:           flag.Int("warmup", -1, "Warmup epochs override"),
		epochs:           flag.Int("epochs", -1, "Timed epochs override"),
		keepAlive:        flag.Float64("keep-alive", -1, "Keep alive duration in seconds (-1 keeps model loaded)"),
		seed:             flag.Int("seed", 42, "Seed"),
		temperature:      flag.Float64("temperature", 0, "Temperature"),
		timeout:          flag.Int("timeout", 1800, "Timeout in seconds (0 disables timeout)"),
		stream:           flag.Bool("stream", true, "Use streaming responses"),
		failFast:         flag.Bool("fail-fast", false, "Stop entire sweep on first failure"),
		probeIntervalMS:  flag.Int("probe-interval-ms", 100, "GPU probe interval in milliseconds"),
		output:           flag.String("output", "", "CSV summary output path"),
		jsonlOutput:      flag.String("jsonl-output", "", "JSONL output path"),
		summaryOutput:    flag.String("summary-output", "", "Text summary output path"),
		outputDir:        flag.String("output-dir", "/results", "Output directory for default outputs"),
		captureOllamaPS:  flag.Bool("capture-ollama-ps", true, "Capture `ollama ps` before and after runs when available"),
		captureGPU:       flag.Bool("capture-gpu", true, "Capture GPU metrics via NVML or nvidia-smi"),
		captureRunnerRSS: flag.Bool("capture-runner-rss", true, "Capture runner RSS when possible"),
		debug:            flag.Bool("debug", false, "Enable debug logging"),
		progress:         flag.String("progress", "auto", "Progress display mode [auto|on|off]"),
		progressWidth:    flag.Int("progress-width", 30, "Progress bar width"),
	}
}

func loadConfig(r rawFlags) (config, error) {
	if strings.TrimSpace(*r.hosts) == "" {
		return config{}, errors.New("--hosts is required")
	}
	if strings.TrimSpace(*r.model) == "" {
		return config{}, errors.New("--model is required")
	}

	profile := strings.ToLower(strings.TrimSpace(*r.profile))
	if !slices.Contains([]string{"quick", "full", "staircase", "impact", "regression", "turbo-benefit", "capacity", "spill"}, profile) {
		return config{}, fmt.Errorf("invalid profile %q", *r.profile)
	}

	hostURLs, err := parseCSV(*r.hosts)
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

	kvModeValue := *r.kvModes
	if strings.TrimSpace(kvModeValue) == "" {
		kvModeValue = strings.Join(defaultKVModes(profile), ",")
	}
	kvModes, err := parseOptionalCSV(kvModeValue)
	if err != nil {
		return config{}, fmt.Errorf("invalid --kv-modes: %w", err)
	}
	if len(kvModes) == 0 {
		return config{}, errors.New("--kv-modes must contain at least one mode")
	}
	for i := range kvModes {
		kvModes[i], err = normalizeKVMode(kvModes[i])
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

	warmup := profileWarmup(profile)
	if *r.warmup >= 0 {
		warmup = *r.warmup
	}
	epochs := profileEpochs(profile)
	if *r.epochs >= 0 {
		epochs = *r.epochs
	}

	cfg := config{
		Hosts:            hosts,
		Model:            *r.model,
		KVModes:          kvModes,
		Profile:          profile,
		Workloads:        workloads,
		NumCtx:           numCtx,
		Concurrency:      concurrency,
		PromptTokens:     *r.promptTokens,
		MaxTokens:        *r.maxTokens,
		Warmup:           warmup,
		Epochs:           epochs,
		KeepAlive:        time.Duration(*r.keepAlive * float64(time.Second)),
		Seed:             *r.seed,
		Temperature:      *r.temperature,
		Timeout:          clientTimeout,
		Stream:           *r.stream,
		FailFast:         *r.failFast,
		ProbeInterval:    time.Duration(*r.probeIntervalMS) * time.Millisecond,
		CaptureOllamaPS:  *r.captureOllamaPS,
		CaptureGPU:       *r.captureGPU,
		CaptureRunnerRSS: *r.captureRunnerRSS,
		OutputDir:        *r.outputDir,
		Debug:            *r.debug,
		ProgressWidth:    *r.progressWidth,
	}

	mode, err := parseProgressMode(*r.progress)
	if err != nil {
		return config{}, err
	}
	cfg.ProgressMode = mode

	cfg.OutputPath = *r.output
	cfg.JSONLPath = *r.jsonlOutput
	cfg.SummaryPath = *r.summaryOutput
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

func parseWorkloads(value string) ([]workloadName, error) {
	parts, err := parseCSV(value)
	if err != nil {
		return nil, err
	}
	out := make([]workloadName, 0, len(parts))
	for _, part := range parts {
		w := workloadName(strings.ToLower(strings.TrimSpace(part)))
		switch w {
		case workloadPrefillHeavy, workloadDecodeGrowth, workloadParallelAmplifier, workloadNearOOMStaircase:
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
		cfg.SummaryPath = filepath.Join(outputDir, "kvstress_summary.txt")
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
	default:
		return 6
	}
}

func defaultKVModes(profile string) []string {
	switch profile {
	case "regression":
		return []string{"f16"}
	case "turbo-benefit", "capacity", "spill":
		return []string{"f16", "tq25", "tq35"}
	default:
		return []string{"f16", "q8_0", "q4_0", "tq25", "tq35"}
	}
}

func ensureOutputDirs(cfg config) error {
	dirs := []string{
		filepath.Dir(cfg.OutputPath),
		filepath.Dir(cfg.JSONLPath),
		filepath.Dir(cfg.SummaryPath),
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
