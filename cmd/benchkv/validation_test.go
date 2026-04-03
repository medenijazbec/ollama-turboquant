package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestRunValidationScaffoldsNonRecallHooks(t *testing.T) {
	cfg := config{Model: "m"}
	cell := sweepCell{
		Host:     hostTarget{},
		KVMode:   "tq35",
		Workload: workloadSpec{Name: workloadPrefillHeavy, NumCtx: 4096, MaxTokens: 128},
	}
	row := workerResult{Workload: string(workloadPrefillHeavy), WorkerIndex: 0}
	got := runValidation(cfg, cell, row)
	if got.Kind != validationLongContext || got.Status != validationScaffolded {
		t.Fatalf("unexpected validation result: %+v", got)
	}
}

func TestRunValidationRecallDistancePasses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/generate":
			w.Header().Set("Content-Type", "application/x-ndjson")
			_ = json.NewEncoder(w).Encode(map[string]any{"response": "TQ-NEEDLE-4317", "done": true})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	parsed, _ := url.Parse(server.URL)
	cfg := config{Model: "m"}
	cell := sweepCell{
		Host:     hostTarget{BaseURL: server.URL, Label: "turbo", URL: parsed, Client: api.NewClient(parsed, server.Client()), KVSupportMode: hostKVSupportRequest},
		KVMode:   "tq35",
		Workload: workloadSpec{Name: workloadDecodeGrowth, NumCtx: 4096, MaxTokens: 16},
	}
	row := workerResult{Workload: string(workloadDecodeGrowth), WorkerIndex: 0}

	got := runValidation(cfg, cell, row)
	if got.Kind != validationRecallDistance || got.Status != validationPassed {
		t.Fatalf("unexpected validation result: %+v", got)
	}
	if !strings.Contains(got.Observed, "TQ-NEEDLE-4317") {
		t.Fatalf("expected observed output to contain recall needle, got %q", got.Observed)
	}
}

func TestNormalizeValidationText(t *testing.T) {
	got := normalizeValidationText("  Hello \n WORLD  ")
	if got != "hello world" {
		t.Fatalf("normalizeValidationText() = %q, want %q", got, "hello world")
	}
}

func TestMinValidationTimeout(t *testing.T) {
	if got := minValidationTimeout(0); got <= 0 {
		t.Fatalf("expected default timeout, got %v", got)
	}
	if got := minValidationTimeout(5 * time.Second); got != 5*time.Second {
		t.Fatalf("expected passthrough timeout, got %v", got)
	}
}

func TestRunValidationSkipsWarmupAndNonPrimaryWorkers(t *testing.T) {
	got := runValidation(config{}, sweepCell{}, workerResult{Warmup: true, WorkerIndex: 0})
	if got.Status != validationSkipped {
		t.Fatalf("expected warmup to skip validation, got %+v", got)
	}
	got = runValidation(config{}, sweepCell{}, workerResult{Warmup: false, WorkerIndex: 1})
	if got.Status != validationSkipped {
		t.Fatalf("expected non-primary worker to skip validation, got %+v", got)
	}
}

func TestRunRecallDistanceValidationFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_ = json.NewEncoder(w).Encode(map[string]any{"response": "wrong-answer", "done": true})
	}))
	defer server.Close()

	parsed, _ := url.Parse(server.URL)
	cfg := config{Model: "m"}
	cell := sweepCell{
		Host:     hostTarget{BaseURL: server.URL, Label: "turbo", URL: parsed, Client: api.NewClient(parsed, server.Client()), KVSupportMode: hostKVSupportRequest},
		KVMode:   "tq35",
		Workload: workloadSpec{Name: workloadDecodeGrowth, NumCtx: 4096, MaxTokens: 16},
	}
	got := runRecallDistanceValidation(cfg, cell)
	if got.Status != validationFailed {
		t.Fatalf("expected failed validation, got %+v", got)
	}
}

func TestRunValidationLongContextRecallPasses(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		_ = json.NewEncoder(w).Encode(map[string]any{"response": "string=LC-NEEDLE-7781 number=48291057", "done": true})
	}))
	defer server.Close()

	parsed, _ := url.Parse(server.URL)
	cfg := config{Model: "m"}
	cell := sweepCell{
		Host:     hostTarget{BaseURL: server.URL, Label: "turbo", URL: parsed, Client: api.NewClient(parsed, server.Client()), KVSupportMode: hostKVSupportRequest},
		KVMode:   "q8_0/tq35",
		Workload: workloadSpec{Name: workloadLongContextRecall, NumCtx: 128000, MaxTokens: 64},
	}
	got := runLongContextRecallValidation(cfg, cell)
	if got.Kind != validationLongContextRecall || got.Status != validationPassed {
		t.Fatalf("unexpected validation result: %+v", got)
	}
}

func TestRunPromptFileRegressionValidationUsesFileFixture(t *testing.T) {
	dir := t.TempDir()
	promptPath := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(promptPath, []byte(strings.Repeat("loaded from file ", 40)), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_ = json.NewEncoder(w).Encode(map[string]any{"response": "stable summary", "done": true, "prompt": body["prompt"]})
	}))
	defer server.Close()

	parsed, _ := url.Parse(server.URL)
	cfg := config{Model: "m", PromptFile: promptPath}
	cell := sweepCell{
		Host:     hostTarget{BaseURL: server.URL, Label: "turbo", URL: parsed, Client: api.NewClient(parsed, server.Client()), KVSupportMode: hostKVSupportRequest},
		KVMode:   "tq35",
		Workload: workloadSpec{Name: workloadPromptFileRegress, NumCtx: 64000, MaxTokens: 64},
	}
	got := runPromptFileRegressionValidation(cfg, cell)
	if got.Kind != validationPromptFileRegression || got.Status != validationPassed {
		t.Fatalf("unexpected validation result: %+v", got)
	}
}

func TestDetectCorruptionMarkersFlagsDegenerateOutput(t *testing.T) {
	longPromptLikeOutput := strings.Repeat("token ", 300) + " ???? //// {\"answer\":"
	markers := detectCorruptionMarkers(longPromptLikeOutput)
	if len(markers) == 0 {
		t.Fatal("expected corruption markers")
	}
}

func TestLongJSONRetentionRemainsScaffolded(t *testing.T) {
	got := runValidation(config{}, sweepCell{}, workerResult{Workload: string(workloadLongJSONRetention), WorkerIndex: 0})
	if got.Kind != validationLongJSONRetention || got.Status != validationScaffolded {
		t.Fatalf("unexpected validation result: %+v", got)
	}
}
