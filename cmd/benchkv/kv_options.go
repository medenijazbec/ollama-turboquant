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

	switch host.KVSupportMode {
	case hostKVSupportLegacy:
		if kvMode != "f16" {
			disposition.Supported = false
			disposition.Error = legacyKVUnsupportedError(kvMode)
			return nil, disposition
		}
		return options, disposition
	case hostKVSupportRequest:
		options["kv_cache_type"] = kvMode
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
