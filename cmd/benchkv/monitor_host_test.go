package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestPreflightHostLegacyOmitsKVOptionsAndMarksUnsupported(t *testing.T) {
	var generateBodies []map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/version":
			_ = json.NewEncoder(w).Encode(map[string]any{"version": "0.18.3"})
		case "/api/show":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"details": map[string]any{
					"family":             "qwen3moe",
					"quantization_level": "Q4_K_M",
				},
			})
		case "/api/generate":
			var body map[string]any
			_ = json.NewDecoder(r.Body).Decode(&body)
			generateBodies = append(generateBodies, body)
			w.Header().Set("Content-Type", "application/x-ndjson")
			_, _ = w.Write([]byte("{\"done\":true,\"prompt_eval_count\":128,\"eval_count\":1}\n"))
		case "/api/ps":
			_ = json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	parsed, _ := url.Parse(server.URL)
	host := hostTarget{
		BaseURL:       server.URL,
		Label:         "baseline",
		URL:           parsed,
		Client:        api.NewClient(parsed, server.Client()),
		KVSupportMode: hostKVSupportLegacy,
	}
	cfg := config{
		Model:   "m",
		KVModes: []string{"f16", "tq35"},
		Timeout: 5 * time.Second,
	}

	preflight, err := preflightHost(context.Background(), cfg, host)
	if err != nil {
		t.Fatalf("preflightHost failed: %v", err)
	}
	if len(generateBodies) != 1 {
		t.Fatalf("generate count = %d, want 1", len(generateBodies))
	}
	options, _ := generateBodies[0]["options"].(map[string]any)
	if _, ok := options["kv_cache_type"]; ok {
		t.Fatal("legacy preflight should not send kv_cache_type")
	}
	if !preflight.KVSupport["f16"].Supported {
		t.Fatal("expected legacy f16 to be supported")
	}
	if preflight.KVSupport["tq35"].Supported {
		t.Fatal("expected legacy tq35 to be unsupported")
	}
	if preflight.KVSupport["tq35"].Status != statusUnsupported {
		t.Fatalf("unexpected legacy status: %q", preflight.KVSupport["tq35"].Status)
	}
	if !strings.Contains(preflight.KVSupport["tq35"].Error, "unsupported kv_cache_type") {
		t.Fatalf("unexpected legacy error: %q", preflight.KVSupport["tq35"].Error)
	}
}

func TestRunProbeRequestRequestHostIncludesKVOptions(t *testing.T) {
	var generateBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&generateBody)
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"done\":true,\"prompt_eval_count\":128,\"eval_count\":1}\n"))
	}))
	defer server.Close()

	parsed, _ := url.Parse(server.URL)
	host := hostTarget{
		BaseURL:       server.URL,
		Label:         "turbo",
		URL:           parsed,
		Client:        api.NewClient(parsed, server.Client()),
		KVSupportMode: hostKVSupportRequest,
	}

	if _, err := runProbeRequest(context.Background(), host, "m", "tq35", 4096, 128, 5*time.Second); err != nil {
		t.Fatalf("runProbeRequest failed: %v", err)
	}

	options, _ := generateBody["options"].(map[string]any)
	if got := options["kv_cache_type"]; got != "tq35" {
		t.Fatalf("kv_cache_type = %#v, want tq35", got)
	}
	if got := options["kv_cache_backend"]; got != "cuda" {
		t.Fatalf("kv_cache_backend = %#v, want cuda", got)
	}
}

func TestHostMetricsStatsReportUnavailableSource(t *testing.T) {
	monitor := newHostMetricsMonitor(0, true)
	stats := monitor.stats()
	if stats.Available {
		t.Fatalf("expected unavailable stats, got %+v", stats)
	}
	if stats.Source != "unavailable" {
		t.Fatalf("Source = %q, want unavailable", stats.Source)
	}
	if stats.HostRAMUsedBytes != nil || stats.ProcessRSSBytes != nil {
		t.Fatalf("expected nil unavailable metrics, got %+v", stats)
	}
}

func TestHostMetricsStatsKeepSourceAndNilProcessRSS(t *testing.T) {
	monitor := newHostMetricsMonitor(0, true)
	monitor.available = true
	monitor.source = "proc-meminfo"
	monitor.baselineRAM = 1024
	monitor.currentRAM = 2048
	monitor.peakRAM = 4096
	monitor.samples = 2

	stats := monitor.stats()
	if !stats.Available || stats.Source != "proc-meminfo" {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if stats.ProcessRSSBytes != nil {
		t.Fatalf("expected nil process rss, got %+v", stats)
	}
	if stats.PeakHostRAMDeltaBytes == nil || *stats.PeakHostRAMDeltaBytes != 3072 {
		t.Fatalf("unexpected peak host delta: %+v", stats)
	}
}
