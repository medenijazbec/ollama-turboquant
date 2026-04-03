package main

import "strings"

func splitBenchmarkKVMode(mode string) (string, string, bool) {
	mode = strings.ToLower(strings.TrimSpace(mode))
	if mode == "" {
		return "f16", "f16", true
	}
	if !strings.Contains(mode, "/") {
		return mode, mode, true
	}
	parts := strings.Split(mode, "/")
	if len(parts) != 2 {
		return "", "", false
	}
	kType := strings.TrimSpace(parts[0])
	vType := strings.TrimSpace(parts[1])
	if kType == "" || vType == "" {
		return "", "", false
	}
	return kType, vType, true
}

func summarizeBenchmarkKVMode(mode string) string {
	kType, vType, ok := splitBenchmarkKVMode(mode)
	if !ok {
		return ""
	}
	return summarizeRequestedOrEffectiveMode(kType, vType)
}

func benchmarkModeUsesTurbo(mode string) bool {
	kType, vType, ok := splitBenchmarkKVMode(mode)
	if !ok {
		return false
	}
	return isTurboQuantMode(kType) || isTurboQuantMode(vType)
}
