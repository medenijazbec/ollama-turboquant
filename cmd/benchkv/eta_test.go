package main

import (
	"testing"
	"time"
)

func TestEstimateStageETALoadingLowConfidence(t *testing.T) {
	eta, confidence, progressKnown := estimateStageETA("loading", 0, 0, 0, 0, 0, 0, time.Second)
	if eta != nil {
		t.Fatalf("expected nil ETA for loading, got %v", eta)
	}
	if confidence != "low" {
		t.Fatalf("confidence = %q, want low", confidence)
	}
	if progressKnown {
		t.Fatal("expected loading progress to be unknown")
	}
}

func TestEstimateStageETADecodeKnown(t *testing.T) {
	eta, confidence, progressKnown := estimateStageETA("decode", 0, 0, 50, 100, 0, 25, time.Second)
	if eta == nil {
		t.Fatal("expected ETA for decode stage")
	}
	if confidence != "high" {
		t.Fatalf("confidence = %q, want high", confidence)
	}
	if !progressKnown {
		t.Fatal("expected decode progress to be known")
	}
}
