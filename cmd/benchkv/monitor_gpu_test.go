package main

import "testing"

func TestParseNvidiaSMI(t *testing.T) {
	out := "1000, 50\n2000, 70\n"
	peakVRAM, meanUtil, ok := parseNvidiaSMI(out)
	if !ok {
		t.Fatal("expected parse to succeed")
	}
	if peakVRAM != int64(3000*1024*1024) {
		t.Fatalf("unexpected vram bytes: %d", peakVRAM)
	}
	if meanUtil != 60 {
		t.Fatalf("unexpected mean util: %.2f", meanUtil)
	}
}
