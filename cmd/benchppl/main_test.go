package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestLoadCorpusChunks(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "corpus.txt")
	if err := os.WriteFile(path, []byte("one two three four five six seven eight"), 0o644); err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}

	chunks, err := loadCorpusChunks(path, 3, 3)
	if err != nil {
		t.Fatalf("loadCorpusChunks failed: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunk count = %d, want 3", len(chunks))
	}
	if chunks[0] != "one two three" {
		t.Fatalf("unexpected first chunk: %q", chunks[0])
	}
}

func TestApplyBaselinePPLDeltas(t *testing.T) {
	rows := []aggregateResult{
		{
			HostLabel:               "baseline",
			Model:                   "qwen3.5-9b:udq4kxl",
			RequestedCacheTypeK:     "f16",
			RequestedCacheTypeV:     "f16",
			FlashAttentionRequested: false,
			Perplexity:              10,
			Status:                  "ok",
		},
		{
			HostLabel:               "turbo",
			Model:                   "qwen3.5-9b:udq4kxl",
			RequestedCacheTypeK:     "tq35",
			RequestedCacheTypeV:     "tq35",
			FlashAttentionRequested: false,
			Perplexity:              12.5,
			Status:                  "ok",
		},
	}

	applyBaselinePPLDeltas(rows)
	if rows[1].PPLDeltaVsBaseline == nil || *rows[1].PPLDeltaVsBaseline != 2.5 {
		t.Fatalf("unexpected ppl delta: %+v", rows[1].PPLDeltaVsBaseline)
	}
}

func TestSplitChunkPromptTarget(t *testing.T) {
	prompt, target, ok := splitChunkPromptTarget("one two three four five six seven eight")
	if !ok {
		t.Fatal("expected prompt/target split")
	}
	if prompt == "" || target == "" {
		t.Fatalf("expected non-empty prompt and target, got %q / %q", prompt, target)
	}
}

func TestComputeSparseKLD(t *testing.T) {
	baseline := []api.Logprob{
		{TokenLogprob: api.TokenLogprob{Token: "a", Logprob: -0.1}, TopLogprobs: []api.TokenLogprob{{Token: "b", Logprob: -2.0}}},
	}
	candidate := []api.Logprob{
		{TokenLogprob: api.TokenLogprob{Token: "a", Logprob: -0.2}, TopLogprobs: []api.TokenLogprob{{Token: "b", Logprob: -1.8}}},
	}
	kld, ok := computeSparseKLD(baseline, candidate)
	if !ok {
		t.Fatal("expected sparse KLD to be available")
	}
	if kld < 0 {
		t.Fatalf("expected non-negative KLD, got %v", kld)
	}
}

func TestParseQJLModes(t *testing.T) {
	if got := parseQJLModes("off"); len(got) != 1 || got[0] {
		t.Fatalf("parseQJLModes(off) = %#v, want [false]", got)
	}
	if got := parseQJLModes("on"); len(got) != 1 || !got[0] {
		t.Fatalf("parseQJLModes(on) = %#v, want [true]", got)
	}
	if got := parseQJLModes("both"); len(got) != 2 || got[0] || !got[1] {
		t.Fatalf("parseQJLModes(both) = %#v, want [false true]", got)
	}
}

func TestApplyPPLLiveSummary(t *testing.T) {
	row := chunkResult{}
	value := 12.5
	applyPPLLiveSummary(&row, pplLiveSummary{
		TelemetryPath:        "telemetry/test.jsonl",
		AvgPromptTPS:         &value,
		ETAConfidence:        "low",
		ProgressSamples:      3,
		LiveStatusEnabled:    true,
		TelemetrySamplingSec: 1,
	})
	if row.TelemetryPathJSONL == "" || row.AvgPromptTPS == nil || *row.AvgPromptTPS != value {
		t.Fatalf("summary not applied: %+v", row)
	}
}

func TestChunkResultJSONIncludesTelemetryPath(t *testing.T) {
	data, err := json.Marshal(chunkResult{TelemetryPathJSONL: "telemetry/test.jsonl"})
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}
	if string(data) == "{}" {
		t.Fatalf("expected telemetry path in json, got %s", string(data))
	}
}
