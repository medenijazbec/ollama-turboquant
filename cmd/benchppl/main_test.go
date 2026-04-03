package main

import (
	"os"
	"path/filepath"
	"testing"
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
