package main

import "testing"

func TestClassifyContextFailure(t *testing.T) {
	tests := []struct {
		stage   string
		message string
		want    string
	}{
		{stage: "load", message: "CUDA error: out of memory", want: "vram_exhausted"},
		{stage: "prefill", message: "cannot allocate memory", want: "host_ram_exhausted"},
		{stage: "timeout", message: "context ladder timeout", want: "timeout"},
		{stage: "backend_refusal", message: "unsupported context length", want: "runtime_context_cap"},
	}
	for _, tt := range tests {
		if got := classifyContextFailure(tt.stage, tt.message); got != tt.want {
			t.Fatalf("classifyContextFailure(%q, %q) = %q, want %q", tt.stage, tt.message, got, tt.want)
		}
	}
}

func TestFormatRejectedRungs(t *testing.T) {
	got := formatRejectedRungs([]ladderRejection{
		{NumCtx: 1000000, Stage: "load", Class: "vram_exhausted"},
		{NumCtx: 262144, Stage: "decode", Class: "validation_failure"},
	})
	if got != "1000000:load:vram_exhausted;262144:decode:validation_failure" {
		t.Fatalf("unexpected rejected rung summary: %q", got)
	}
}
