package main

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestTelemetryWriterWritesJSONL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sample.telemetry.jsonl")
	writer, err := newTelemetryWriter(path)
	if err != nil {
		t.Fatalf("newTelemetryWriter failed: %v", err)
	}
	if err := writer.write(liveSample{Suite: "suite", Stage: "loading", Timestamp: time.Now().UTC()}); err != nil {
		t.Fatalf("write failed: %v", err)
	}
	if err := writer.close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	count := 0
	for scanner.Scan() {
		count++
	}
	if count != 1 {
		t.Fatalf("sample count = %d, want 1", count)
	}
}
