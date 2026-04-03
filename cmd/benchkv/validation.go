package main

import (
	"context"
	"errors"
	"os"
	"strings"
	"time"

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

	validationPassed     validationStatus = "passed"
	validationFailed     validationStatus = "failed"
	validationScaffolded validationStatus = "scaffolded"
	validationSkipped    validationStatus = "skipped"
)

type validationResult struct {
	Kind     validationKind
	Status   validationStatus
	Expected string
	Observed string
	Error    string
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
	case workloadPromptFileRegress:
		return runPromptFileRegressionValidation(cfg, cell)
	case workloadDecodeCorruption:
		return runDecodeCorruptionValidation(cfg, cell)
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
	markers := detectCorruptionMarkers(observed)
	if len(markers) == 0 && strings.Contains(observed, "12345") {
		return validationResult{Kind: validationDecodeCorruptionGuard, Status: validationPassed, Expected: "valid JSON answer with checksum 12345", Observed: observed}
	}
	return validationResult{Kind: validationDecodeCorruptionGuard, Status: validationFailed, Expected: "valid JSON answer with checksum 12345", Observed: observed, Error: strings.Join(markers, ",")}
}

func runValidationGenerate(cfg config, cell sweepCell, prompt string, maxTokens int) (string, error) {
	stream := false
	keepAlive := api.Duration{Duration: cfg.KeepAlive}
	options, disposition := buildGenerateOptions(cell.Host, cell.KVMode, cell.Workload.NumCtx, maxTokens, 0, 0)
	if !disposition.Supported {
		return "", errors.New(disposition.Error)
	}

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
	lower := strings.ToLower(observed)
	var markers []string
	if strings.Contains(lower, "////") || strings.Contains(lower, "????") {
		markers = append(markers, "slash-question repetition")
	}
	if strings.Count(lower, "{") != strings.Count(lower, "}") {
		markers = append(markers, "malformed-json-braces")
	}
	if strings.Contains(lower, `""""`) || strings.Contains(lower, `,,,,`) {
		markers = append(markers, "degenerate repetition")
	}
	if repeatedTokenRun(lower, 12) {
		markers = append(markers, "repeated-token run")
	}
	return markers
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

func normalizeValidationText(value string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(value))), " ")
}

func minValidationTimeout(runTimeout time.Duration) time.Duration {
	if runTimeout <= 0 || runTimeout > 30*time.Second {
		return 30 * time.Second
	}
	return runTimeout
}
