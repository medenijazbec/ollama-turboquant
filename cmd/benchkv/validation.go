package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
)

type validationKind string
type validationStatus string

const (
	validationRecallDistance        validationKind = "recall-distance"
	validationPerplexity            validationKind = "perplexity"
	validationLongContext           validationKind = "long-context"
	validationStructuredOutput      validationKind = "structured-output"
	validationLongContextRecall     validationKind = "long-context-recall"
	validationLongJSONRetention     validationKind = "long-json-retention"
	validationPromptFileRegression  validationKind = "prompt-file-regression"
	validationDecodeCorruptionGuard validationKind = "decode-corruption-guard"
	validationFitCeiling            validationKind = "fit-ceiling"
	validationNIAHRetrieval         validationKind = "niah-retrieval"

	validationPassed     validationStatus = "passed"
	validationFailed     validationStatus = "failed"
	validationScaffolded validationStatus = "scaffolded"
	validationSkipped    validationStatus = "skipped"
)

type validationResult struct {
	Kind               validationKind
	Status             validationStatus
	Expected           string
	Observed           string
	Error              string
	CorruptionClass    string
	EmptyOutput        bool
	TruncationDetected bool
	NIAHDepth          int
	NIAHPass           bool
}

// Implemented benchmark-visible correctness hooks so path diagnostics are not judged on token speed alone; idea source: @TheTom.
func runValidation(cfg config, cell sweepCell, row workerResult) validationResult {
	if row.Warmup || row.WorkerIndex != 0 {
		return validationResult{Kind: validationRecallDistance, Status: validationSkipped}
	}
	switch workloadName(row.Workload) {
	case workloadDecodeGrowth:
		// Implemented a concrete recall-under-distance validation path to catch context-pressure degradation; idea source: @primoco.
		return runRecallDistanceValidation(cfg, cell)
	case workloadLongContextRecall:
		return runLongContextRecallValidation(cfg, cell)
	case workloadNIAHRetrieval:
		return runNIAHRetrievalValidation(cfg, cell)
	case workloadPromptFileRegress:
		return runPromptFileRegressionValidation(cfg, cell)
	case workloadDecodeCorruption:
		return runDecodeCorruptionValidation(cfg, cell)
	case workloadAgenticStructured:
		// Implemented agentic and long-context validation in the benchmark harness so we do not judge TurboQuant on PPL or speed alone; idea source: @seanrasch.
		return runAgenticStructuredValidation(cfg, cell)
	case workloadLongJSONRetention:
		return validationResult{Kind: validationLongJSONRetention, Status: validationScaffolded, Error: "long-json retention scaffolded; no concrete implementation yet"}
	case workloadFitCeiling:
		return validationResult{Kind: validationFitCeiling, Status: validationSkipped}
	case workloadPrefillHeavy:
		// Implemented scaffolded long-context and structured-output validation hooks so those checks have explicit ownership in the benchmark path; idea source: @seanrasch.
		return validationResult{Kind: validationLongContext, Status: validationScaffolded, Error: "long-context validation scaffolded; no concrete implementation yet"}
	case workloadParallelAmplifier:
		return validationResult{Kind: validationStructuredOutput, Status: validationScaffolded, Error: "structured-output validation scaffolded; no concrete implementation yet"}
	default:
		// Implemented explicit validation result states so corruption-sensitive failures are not conflated with throughput success; idea source: @sjoerdmaessen.
		return validationResult{Kind: validationPerplexity, Status: validationScaffolded, Error: "perplexity harness scaffolded; no concrete implementation yet"}
	}
}

type agenticStructuredResult struct {
	Winner     string  `json:"winner"`
	Confidence float64 `json:"confidence"`
	Conflict   bool    `json:"conflict"`
}

func runLongContextRecallValidation(cfg config, cell sweepCell) validationResult {
	const expectedString = "LC-NEEDLE-7781"
	const expectedNumber = "48291057"
	prompt := buildLongContextRecallPrompt(max(cell.Workload.PromptTokensTarget, 4096))
	observed, err := runValidationGenerate(cfg, cell, prompt, 64)
	if err != nil {
		return validationResult{Kind: validationLongContextRecall, Status: validationFailed, Expected: "string=" + expectedString + " number=" + expectedNumber, Observed: observed, Error: err.Error()}
	}
	normalized := normalizeValidationText(observed)
	if strings.Contains(normalized, normalizeValidationText(expectedString)) && strings.Contains(normalized, normalizeValidationText(expectedNumber)) {
		return validationResult{Kind: validationLongContextRecall, Status: validationPassed, Expected: "string=" + expectedString + " number=" + expectedNumber, Observed: observed}
	}
	return validationResult{Kind: validationLongContextRecall, Status: validationFailed, Expected: "string=" + expectedString + " number=" + expectedNumber, Observed: observed, Error: "long-context recall response did not retain both needles"}
}

func runPromptFileRegressionValidation(cfg config, cell sweepCell) validationResult {
	inlinePrompt := buildPromptFileRegressionPrompt(max(cell.Workload.PromptTokensTarget, 320))
	inlineObserved, inlineErr := runValidationGenerate(cfg, cell, inlinePrompt, 96)
	fileObserved := inlineObserved
	fileErr := inlineErr
	if strings.TrimSpace(cfg.PromptFile) != "" {
		data, err := os.ReadFile(cfg.PromptFile)
		if err != nil {
			return validationResult{Kind: validationPromptFileRegression, Status: validationFailed, Expected: "file-backed prompt should complete without corruption", Observed: "", Error: err.Error()}
		}
		fileObserved, fileErr = runValidationGenerate(cfg, cell, string(data), 96)
	}
	if inlineErr != nil {
		return validationResult{Kind: validationPromptFileRegression, Status: validationFailed, Expected: "inline prompt should complete", Observed: inlineObserved, Error: inlineErr.Error()}
	}
	if fileErr != nil {
		return validationResult{Kind: validationPromptFileRegression, Status: validationFailed, Expected: "file-backed prompt should complete", Observed: fileObserved, Error: fileErr.Error()}
	}
	inlineMarkers := detectCorruptionMarkers(inlineObserved)
	fileMarkers := detectCorruptionMarkers(fileObserved)
	for attempt := 1; attempt < cfg.Repeats; attempt++ {
		repeatObserved, repeatErr := runValidationGenerate(cfg, cell, inlinePrompt, 96)
		if repeatErr != nil {
			return validationResult{Kind: validationPromptFileRegression, Status: validationFailed, Expected: "repeated prompt should complete", Observed: repeatObserved, Error: repeatErr.Error()}
		}
		inlineMarkers = append(inlineMarkers, detectCorruptionMarkers(repeatObserved)...)
	}
	if len(inlineMarkers) == 0 && len(fileMarkers) == 0 {
		return validationResult{Kind: validationPromptFileRegression, Status: validationPassed, Expected: "no prompt-ingestion corruption markers", Observed: "inline=" + inlineObserved + " | file=" + fileObserved}
	}
	return validationResult{Kind: validationPromptFileRegression, Status: validationFailed, Expected: "no prompt-ingestion corruption markers", Observed: "inline=" + inlineObserved + " | file=" + fileObserved, Error: strings.Join(append(inlineMarkers, fileMarkers...), ",")}
}

func runDecodeCorruptionValidation(cfg config, cell sweepCell) validationResult {
	observed, err := runValidationGenerate(cfg, cell, buildDecodeCorruptionPrompt(max(cell.Workload.PromptTokensTarget, 1024)), max(cell.Workload.MaxTokens, 2048))
	if err != nil {
		return validationResult{Kind: validationDecodeCorruptionGuard, Status: validationFailed, Expected: "valid JSON answer with checksum 12345", Observed: observed, Error: err.Error()}
	}
	markers, corruptionClass, emptyOutput, truncationDetected := classifyCorruption(observed)
	for attempt := 1; attempt < cfg.Repeats; attempt++ {
		repeatObserved, repeatErr := runValidationGenerate(cfg, cell, buildDecodeCorruptionPrompt(max(cell.Workload.PromptTokensTarget, 1024)), max(cell.Workload.MaxTokens, 2048))
		if repeatErr != nil {
			return validationResult{Kind: validationDecodeCorruptionGuard, Status: validationFailed, Expected: "valid JSON answer with checksum 12345", Observed: repeatObserved, Error: repeatErr.Error()}
		}
		repeatMarkers, repeatClass, repeatEmpty, repeatTrunc := classifyCorruption(repeatObserved)
		markers = append(markers, repeatMarkers...)
		if corruptionClass == "" {
			corruptionClass = repeatClass
		}
		emptyOutput = emptyOutput || repeatEmpty
		truncationDetected = truncationDetected || repeatTrunc
	}
	if len(markers) == 0 && strings.Contains(observed, "12345") {
		return validationResult{Kind: validationDecodeCorruptionGuard, Status: validationPassed, Expected: "valid JSON answer with checksum 12345", Observed: observed}
	}
	return validationResult{Kind: validationDecodeCorruptionGuard, Status: validationFailed, Expected: "valid JSON answer with checksum 12345", Observed: observed, Error: strings.Join(markers, ","), CorruptionClass: corruptionClass, EmptyOutput: emptyOutput, TruncationDetected: truncationDetected}
}

func runAgenticStructuredValidation(cfg config, cell sweepCell) validationResult {
	stream := false
	keepAlive := api.Duration{Duration: cfg.KeepAlive}
	options, disposition := buildGenerateOptions(cell.Host, cell.KVMode, cell.Workload.NumCtx, 256, cfg.Seed, 0)
	if !disposition.Supported {
		return validationResult{Kind: validationStructuredOutput, Status: validationSkipped, Error: disposition.Error}
	}
	applyFlashAttentionOption(options, cell.FARequested)
	applyExperimentalTurboQuantOptions(options, cfg, cell)

	req := &api.ChatRequest{
		Model:     cfg.Model,
		Stream:    &stream,
		KeepAlive: &keepAlive,
		Format:    []byte(`{"type":"object","properties":{"winner":{"type":"string"},"confidence":{"type":"number"},"conflict":{"type":"boolean"}},"required":["winner","confidence","conflict"]}`),
		Messages: []api.Message{
			{Role: "user", Content: buildAgenticStructuredPrompt(cfg.ToolSuite)},
		},
		Options: options,
	}

	ctx, cancel := withOptionalTimeout(context.Background(), minValidationTimeout(cfg.Timeout))
	defer cancel()

	var observed strings.Builder
	var finalErr error
	err := cell.Host.Client.Chat(ctx, req, func(resp api.ChatResponse) error {
		observed.WriteString(resp.Message.Content)
		return nil
	})
	if err != nil {
		finalErr = err
	}
	if finalErr != nil {
		return validationResult{Kind: validationStructuredOutput, Status: validationFailed, Expected: `{"winner":"weather_a","confidence":0.82,"conflict":true}`, Observed: observed.String(), Error: finalErr.Error()}
	}
	markers, corruptionClass, emptyOutput, truncationDetected := classifyCorruption(observed.String())
	var parsed agenticStructuredResult
	if len(markers) == 0 && json.Unmarshal([]byte(observed.String()), &parsed) == nil {
		confidenceOK := math.Abs(parsed.Confidence-0.82) <= 0.02
		winnerOK := parsed.Winner == "weather_a" || parsed.Winner == "alpha"
		if winnerOK && parsed.Conflict && confidenceOK {
			return validationResult{Kind: validationStructuredOutput, Status: validationPassed, Expected: `{"winner":"weather_a","confidence":0.82,"conflict":true}`, Observed: observed.String()}
		}
	}
	if len(markers) == 0 {
		markers = append(markers, "semantic-validator mismatch")
	}
	return validationResult{Kind: validationStructuredOutput, Status: validationFailed, Expected: `{"winner":"weather_a","confidence":0.82,"conflict":true}`, Observed: observed.String(), Error: strings.Join(markers, ","), CorruptionClass: corruptionClass, EmptyOutput: emptyOutput, TruncationDetected: truncationDetected}
}

func runValidationGenerate(cfg config, cell sweepCell, prompt string, maxTokens int) (string, error) {
	stream := false
	keepAlive := api.Duration{Duration: cfg.KeepAlive}
	options, disposition := buildGenerateOptions(cell.Host, cell.KVMode, cell.Workload.NumCtx, maxTokens, 0, 0)
	if !disposition.Supported {
		return "", errors.New(disposition.Error)
	}
	applyExperimentalTurboQuantOptions(options, cfg, cell)

	req := &api.GenerateRequest{
		Model:     cfg.Model,
		Prompt:    prompt,
		Raw:       true,
		Stream:    &stream,
		KeepAlive: &keepAlive,
		Options:   options,
	}

	ctx, cancel := withOptionalTimeout(context.Background(), minValidationTimeout(cfg.Timeout))
	defer cancel()

	var observed strings.Builder
	err := cell.Host.Client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		observed.WriteString(resp.Response)
		return nil
	})
	return observed.String(), err
}

func detectCorruptionMarkers(observed string) []string {
	markers, _, _, _ := classifyCorruption(observed)
	return markers
}

func classifyCorruption(observed string) ([]string, string, bool, bool) {
	lower := strings.ToLower(observed)
	var markers []string
	corruptionClass := ""
	emptyOutput := false
	truncationDetected := false
	if strings.TrimSpace(observed) == "" {
		markers = append(markers, "empty-output")
		corruptionClass = "empty_output"
		emptyOutput = true
	}
	if !utf8.ValidString(observed) {
		markers = append(markers, "invalid-utf8")
		if corruptionClass == "" {
			corruptionClass = "invalid_utf8"
		}
	}
	if strings.Contains(lower, "////") || strings.Contains(lower, "????") {
		markers = append(markers, "slash-question repetition")
		if corruptionClass == "" {
			corruptionClass = "punctuation_repetition"
		}
	}
	if strings.Contains(lower, "!!!!!") {
		markers = append(markers, "bang repetition")
		if corruptionClass == "" {
			corruptionClass = "punctuation_repetition"
		}
	}
	if strings.Count(lower, "{") != strings.Count(lower, "}") {
		markers = append(markers, "malformed-json-braces")
		if corruptionClass == "" {
			corruptionClass = "malformed_json"
		}
	}
	if strings.Contains(lower, `""""`) || strings.Contains(lower, `,,,,`) {
		markers = append(markers, "degenerate repetition")
		if corruptionClass == "" {
			corruptionClass = "degenerate_small_alphabet"
		}
	}
	if repeatedTokenRun(lower, 12) {
		markers = append(markers, "repeated-token run")
		if corruptionClass == "" {
			corruptionClass = "degenerate_small_alphabet"
		}
	}
	if strings.HasSuffix(strings.TrimSpace(observed), "{") || strings.HasSuffix(strings.TrimSpace(observed), "[") || strings.HasSuffix(strings.TrimSpace(observed), ",") {
		markers = append(markers, "truncation-detected")
		truncationDetected = true
		if corruptionClass == "" {
			corruptionClass = "truncation_detected"
		}
	}
	if len(markers) == 0 && len(observed) > 0 && strings.Count(lower, "\n") > 40 {
		markers = append(markers, "verbose-runaway")
		if corruptionClass == "" {
			corruptionClass = "verbose_runaway"
		}
	}
	return markers, corruptionClass, emptyOutput, truncationDetected
}

func repeatedTokenRun(text string, threshold int) bool {
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return false
	}
	run := 1
	for i := 1; i < len(fields); i++ {
		if fields[i] == fields[i-1] {
			run++
			if run >= threshold {
				return true
			}
			continue
		}
		run = 1
	}
	return false
}

func runRecallDistanceValidation(cfg config, cell sweepCell) validationResult {
	const expectedNeedle = "TQ-NEEDLE-4317"
	prompt := buildRecallDistancePrompt(expectedNeedle)
	stream := false
	keepAlive := api.Duration{Duration: cfg.KeepAlive}
	options, disposition := buildGenerateOptions(cell.Host, cell.KVMode, cell.Workload.NumCtx, 16, 0, 0)
	if !disposition.Supported {
		return validationResult{Kind: validationRecallDistance, Status: validationSkipped, Error: disposition.Error}
	}
	applyExperimentalTurboQuantOptions(options, cfg, cell)

	req := &api.GenerateRequest{
		Model:     cfg.Model,
		Prompt:    prompt,
		Raw:       true,
		Stream:    &stream,
		KeepAlive: &keepAlive,
		Options:   options,
	}

	ctx, cancel := withOptionalTimeout(context.Background(), minValidationTimeout(cfg.Timeout))
	defer cancel()

	var observed strings.Builder
	var finalErr error
	err := cell.Host.Client.Generate(ctx, req, func(resp api.GenerateResponse) error {
		observed.WriteString(resp.Response)
		return nil
	})
	if err != nil {
		finalErr = err
	}

	observedText := normalizeValidationText(observed.String())
	expectedText := normalizeValidationText(expectedNeedle)
	if finalErr != nil {
		return validationResult{Kind: validationRecallDistance, Status: validationFailed, Expected: expectedNeedle, Observed: observed.String(), Error: finalErr.Error()}
	}
	if strings.Contains(observedText, expectedText) {
		return validationResult{Kind: validationRecallDistance, Status: validationPassed, Expected: expectedNeedle, Observed: observed.String()}
	}
	return validationResult{Kind: validationRecallDistance, Status: validationFailed, Expected: expectedNeedle, Observed: observed.String(), Error: "response did not contain the expected recall needle"}
}

func buildRecallDistancePrompt(needle string) string {
	filler := strings.Repeat("filler-token ", 512)
	return "Memorize the secret token hidden in the middle of the filler text and repeat it exactly at the end.\n" +
		filler + "\nSECRET TOKEN: " + needle + "\n" + filler + "\nReply with the secret token only."
}

func buildAgenticStructuredPrompt(toolSuite string) string {
	return "You are reconciling mock tool outputs. Tool weather_a is the winning tool with confidence 0.82. Tool weather_b reports a lower-confidence alternative at 0.41. The tools disagree, so conflict is true. Return strict JSON with winner, confidence, and conflict only. winner must be the winning tool label. Tool suite: " + firstNonEmpty(toolSuite, "default")
}

func normalizeValidationText(value string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(value))), " ")
}

func minValidationTimeout(runTimeout time.Duration) time.Duration {
	if runTimeout <= 0 {
		return 90 * time.Second
	}
	derived := maxDuration(30*time.Second, runTimeout/4)
	if derived > runTimeout {
		return runTimeout
	}
	return derived
}

func runNIAHRetrievalValidation(cfg config, cell sweepCell) validationResult {
	depth := niahDepthForContext(cell.Workload.NumCtx)
	needle := "NIAH-NEEDLE-314159"
	prompt := buildNIAHPrompt(needle, depth)
	observed, err := runValidationGenerate(cfg, cell, prompt, 24)
	if err != nil {
		return validationResult{Kind: validationNIAHRetrieval, Status: validationFailed, Expected: needle, Observed: observed, Error: err.Error(), NIAHDepth: depth}
	}
	pass := strings.Contains(normalizeValidationText(observed), normalizeValidationText(needle))
	if pass {
		return validationResult{Kind: validationNIAHRetrieval, Status: validationPassed, Expected: needle, Observed: observed, NIAHDepth: depth, NIAHPass: true}
	}
	return validationResult{Kind: validationNIAHRetrieval, Status: validationFailed, Expected: needle, Observed: observed, Error: "needle was not retrieved exactly", NIAHDepth: depth, NIAHPass: false}
}

func niahDepthForContext(numCtx int) int {
	for _, depth := range []int{32768, 16384, 8192, 2048, 512} {
		if numCtx >= depth+512 {
			return depth
		}
	}
	return 512
}

func buildNIAHPrompt(needle string, depth int) string {
	filler := strings.Repeat("hay ", max(1, depth/4))
	return "Read the full context carefully. There is exactly one secret needle. Return the needle only.\n" + filler + "\nNEEDLE: " + needle + "\n" + filler
}

func maxDuration(a, b time.Duration) time.Duration {
	if a > b {
		return a
	}
	return b
}
