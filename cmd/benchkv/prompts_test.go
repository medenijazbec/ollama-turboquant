package main

import "testing"

func TestRenderPromptDeterministic(t *testing.T) {
	a := renderPrompt(8, 0)
	b := renderPrompt(8, 0)
	if a != b {
		t.Fatal("expected deterministic prompt rendering")
	}
}

func TestRenderWorkerPromptVariesOffset(t *testing.T) {
	cal := promptCalibration{WordCount: 8}
	a := renderWorkerPrompt(cal, 1, 0)
	b := renderWorkerPrompt(cal, 1, 1)
	if a == b {
		t.Fatal("expected worker prompts to vary by worker index")
	}
}
