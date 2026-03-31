package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteCSV(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.csv")
	rows := []epochAggregate{{
		Host: "http://127.0.0.1:11438", HostLabel: "baseline", ServerVersion: "0.18.3", Model: "m", KVModeRequested: "f16", KVModeResolved: "f16", KVAlgoResolved: "", KVPath: "dense-fallback",
		Workload: "prefill-heavy", NumCtx: 8192, PromptTokensTarget: 7372, MaxTokens: 128, Concurrency: 1, Epoch: 1, Status: statusOK, Success: boolPtr(true),
	}}
	if err := writeCSV(path, rows); err != nil {
		t.Fatalf("writeCSV failed: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	if !strings.Contains(string(data), "host,host_label,server_version,model") {
		t.Fatal("expected csv header")
	}
	if !strings.Contains(string(data), "quant,kv_mode_requested,kv_mode_resolved") {
		t.Fatal("expected extended csv header")
	}
	if !strings.Contains(string(data), "kv_algo_resolved") {
		t.Fatal("expected kv algorithm csv header")
	}
}
