package main

import "testing"

func TestDefaultHostLabels(t *testing.T) {
	labels := defaultHostLabels(3)
	if labels[0] != "baseline" || labels[1] != "turbo" || labels[2] != "host3" {
		t.Fatalf("unexpected labels: %#v", labels)
	}
}
