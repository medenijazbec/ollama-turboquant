package main

import (
	"context"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

type validationKind string
type validationStatus string

const (
	validationRecallDistance   validationKind = "recall-distance"
	validationPerplexity       validationKind = "perplexity"
	validationLongContext      validationKind = "long-context"
	validationStructuredOutput validationKind = "structured-output"

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
