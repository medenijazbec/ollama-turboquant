package main

import "testing"

func TestBuildGenerateOptionsLegacyF16OmitsKVOptions(t *testing.T) {
	host := hostTarget{KVSupportMode: hostKVSupportLegacy}
	options, disposition := buildGenerateOptions(host, "f16", 8192, 128, 42, 0)
	if !disposition.Supported {
		t.Fatalf("expected supported disposition, got %#v", disposition)
	}
	if _, ok := options["kv_cache_type"]; ok {
		t.Fatal("legacy f16 should not send kv_cache_type")
	}
	if _, ok := options["kv_cache_backend"]; ok {
		t.Fatal("legacy f16 should not send kv_cache_backend")
	}
}

func TestBuildGenerateOptionsLegacyQ80Unsupported(t *testing.T) {
	host := hostTarget{KVSupportMode: hostKVSupportLegacy}
	_, disposition := buildGenerateOptions(host, "q8_0", 8192, 128, 42, 0)
	if disposition.Supported {
		t.Fatalf("expected unsupported disposition, got %#v", disposition)
	}
}

func TestBuildGenerateOptionsRequestF16IncludesKVType(t *testing.T) {
	host := hostTarget{KVSupportMode: hostKVSupportRequest}
	options, disposition := buildGenerateOptions(host, "f16", 8192, 128, 42, 0)
	if !disposition.Supported {
		t.Fatalf("expected supported disposition, got %#v", disposition)
	}
	if got := options["kv_cache_type"]; got != "f16" {
		t.Fatalf("kv_cache_type = %#v, want f16", got)
	}
}

func TestBuildGenerateOptionsRequestTQ35IncludesBackend(t *testing.T) {
	host := hostTarget{KVSupportMode: hostKVSupportRequest}
	options, disposition := buildGenerateOptions(host, "tq35", 8192, 128, 42, 0)
	if !disposition.Supported {
		t.Fatalf("expected supported disposition, got %#v", disposition)
	}
	if got := options["kv_cache_type"]; got != "tq35" {
		t.Fatalf("kv_cache_type = %#v, want tq35", got)
	}
	if got := options["kv_cache_backend"]; got != "cuda" {
		t.Fatalf("kv_cache_backend = %#v, want cuda", got)
	}
}
