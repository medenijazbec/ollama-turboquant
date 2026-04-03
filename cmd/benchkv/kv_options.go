package main

import "fmt"

func buildGenerateOptions(host hostTarget, kvMode string, numCtx int, numPredict int, seed int, temperature float64) (map[string]any, kvRequestDisposition) {
	options := map[string]any{
		"num_ctx":     numCtx,
		"num_predict": numPredict,
		"seed":        seed,
		"temperature": temperature,
	}

	disposition := kvRequestDisposition{Supported: true}
	if kvMode == "" {
		kvMode = "f16"
	}
	kType, vType, ok := splitBenchmarkKVMode(kvMode)
	if !ok {
		disposition.Supported = false
		disposition.Error = fmt.Sprintf("invalid kv mode %q", kvMode)
		return nil, disposition
	}

	switch host.KVSupportMode {
	case hostKVSupportLegacy:
		if kType != "f16" || vType != "f16" {
			disposition.Supported = false
			disposition.Error = legacyKVUnsupportedError(kvMode)
			return nil, disposition
		}
		return options, disposition
	case hostKVSupportRequest:
		if kType == vType {
			options["kv_cache_type"] = kType
		} else {
			options["kv_cache_type_k"] = kType
			options["kv_cache_type_v"] = vType
		}
		disposition.RequestedOverride = true
		if backend := requestedKVBackend(kvMode); backend != "" {
			options["kv_cache_backend"] = backend
		}
		return options, disposition
	default:
		disposition.Supported = false
		disposition.Error = fmt.Sprintf("unknown host kv support mode %q", host.KVSupportMode)
		return nil, disposition
	}
}

func legacyKVUnsupportedError(kvMode string) string {
	return "unsupported kv_cache_type on target host"
}
