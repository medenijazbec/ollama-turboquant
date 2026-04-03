package main

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

func boolPtr(v bool) *bool {
	return &v
}

func int64Ptr(v int64) *int64 {
	return &v
}

func float64Ptr(v float64) *float64 {
	return &v
}

func ptrInt64Value(v *int64) int64 {
	if v == nil {
		return -1
	}
	return *v
}

func ptrFloat64Value(v *float64) float64 {
	if v == nil {
		return -1
	}
	return *v
}

func formatPerGPUVRAMGiB(values map[string]int64) string {
	if len(values) == 0 {
		return ""
	}
	keys := slices.Sorted(maps.Keys(values))
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%.2f", key, float64(values[key])/(1024*1024*1024)))
	}
	return strings.Join(parts, ";")
}
