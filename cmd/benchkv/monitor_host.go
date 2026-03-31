package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

func preflightHost(ctx context.Context, cfg config, host hostTarget) (hostPreflight, error) {
	out := hostPreflight{
		Reachable: true,
		KVSupport: make(map[string]kvSupportResult),
	}

	version, err := fetchVersion(ctx, host)
	if err != nil {
		return out, err
	}
	out.Version = version

	showResp, err := host.Client.Show(ctx, &api.ShowRequest{Model: cfg.Model})
	if err != nil {
		return out, err
	}
	out.ModelFamily = showResp.Details.Family
	out.ModelQuant = showResp.Details.QuantizationLevel

	if cfg.CaptureOllamaPS {
		out.OllamaPSBefore = captureOllamaPS()
	}

	for _, kvMode := range cfg.KVModes {
		result := kvSupportResult{Supported: true, Status: statusOK}
		if host.KVSupportMode == hostKVSupportLegacy {
			if kvMode != "f16" {
				result.Supported = false
				result.Status = statusUnsupported
				result.Error = "unsupported kv_cache_type on target host"
				out.KVSupport[kvMode] = result
				continue
			}
		}

		metrics, err := runProbeRequest(ctx, host, cfg.Model, kvMode, 4096, 128, cfg.Timeout)
		if err != nil {
			if isUnsupportedKVError(err) {
				result.Supported = false
				result.Status = statusUnsupported
				result.Error = "unsupported kv_cache_type on target host"
				out.KVSupport[kvMode] = result
				continue
			} else {
				return out, fmt.Errorf("probe failed for %s %s: %w", host.Label, kvMode, err)
			}
		}
		if metrics == nil {
			return out, fmt.Errorf("probe returned no metrics for %s %s", host.Label, kvMode)
		}
		if _, err := host.Client.ListRunning(ctx); err != nil && cfg.Debug {
			fmt.Printf("warning: %s ListRunning failed during preflight for %s: %v\n", host.Label, kvMode, err)
		}
		out.KVSupport[kvMode] = result
	}

	return out, nil
}

func fetchVersion(ctx context.Context, host hostTarget) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, host.URL.JoinPath("/api/version").String(), nil)
	if err != nil {
		return "", err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		return "", fmt.Errorf("version endpoint returned %s", resp.Status)
	}

	var body struct {
		Version string `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", err
	}
	if body.Version == "" {
		return "unknown", nil
	}
	return body.Version, nil
}

func runProbeRequest(ctx context.Context, host hostTarget, model, kvMode string, numCtx, promptTokens int, timeout time.Duration) (*api.Metrics, error) {
	stream := true
	keepAlive := api.Duration{Duration: -1}
	prompt := renderPrompt(max(1, int(float64(promptTokens)/1.3)), 0)
	options, disposition := buildGenerateOptions(host, kvMode, numCtx, 1, 42, 0)
	if !disposition.Supported {
		return nil, errors.New(disposition.Error)
	}

	req := &api.GenerateRequest{
		Model:     model,
		Prompt:    prompt,
		Raw:       true,
		Stream:    &stream,
		KeepAlive: &keepAlive,
		Options:   options,
	}

	runCtx, cancel := withOptionalTimeout(ctx, timeout)
	defer cancel()

	var metrics *api.Metrics
	err := host.Client.Generate(runCtx, req, func(resp api.GenerateResponse) error {
		if resp.Done {
			copy := resp.Metrics
			metrics = &copy
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if metrics == nil {
		return nil, errors.New("probe did not return final metrics")
	}
	return metrics, nil
}

func captureOllamaPS() string {
	cmd := exec.Command("ollama", "ps")
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func captureRunnerRSS() int64 {
	pgrep := exec.Command("pgrep", "-f", "ollama runner")
	out, err := pgrep.Output()
	if err != nil {
		return -1
	}
	pids := strings.Fields(string(out))
	var maxRSS int64 = -1
	for _, pid := range pids {
		cmd := exec.Command("ps", "-o", "rss=", "-p", pid)
		raw, err := cmd.Output()
		if err != nil {
			continue
		}
		raw = bytes.TrimSpace(raw)
		if len(raw) == 0 {
			continue
		}
		var rssKB int64
		if _, err := fmt.Sscanf(string(raw), "%d", &rssKB); err != nil {
			continue
		}
		rssBytes := rssKB * 1024
		if rssBytes > maxRSS {
			maxRSS = rssBytes
		}
	}
	return maxRSS
}

func isUnsupportedKVError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	needles := []string{
		"unsupported kv_cache_type",
		"unsupported",
		"invalid kv_cache_type",
		"kv cache type",
		"invalid option",
		"unknown field",
		"unknown option",
		"not supported",
	}
	for _, needle := range needles {
		if strings.Contains(msg, needle) {
			return true
		}
	}
	return false
}

func inferGPUResidency(psOutput string) string {
	lower := strings.ToLower(psOutput)
	switch {
	case strings.Contains(lower, "100% gpu"):
		return "100% GPU"
	case strings.Contains(lower, "gpu"):
		return "partial GPU"
	case lower == "":
		return "unknown"
	default:
		return "cpu-or-unknown"
	}
}

func fullGPUResidency(psOutput string) bool {
	return strings.Contains(strings.ToLower(psOutput), "100% gpu")
}

func gpuOffloadRegression(beforePS, afterPS string) bool {
	beforeFull := fullGPUResidency(beforePS)
	afterFull := fullGPUResidency(afterPS)
	if beforeFull && !afterFull {
		return true
	}
	lower := strings.ToLower(afterPS)
	return strings.Contains(lower, "cpu") || strings.Contains(lower, "partial gpu")
}
