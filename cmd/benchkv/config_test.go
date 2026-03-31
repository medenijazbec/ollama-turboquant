package main

import "testing"

func strptr(s string) *string          { return &s }
func intPtr(v int) *int                { return &v }
func floatPtr(v float64) *float64      { return &v }
func testBoolPtr(v bool) *bool         { return &v }

func TestDefaultHostKVSupportModes(t *testing.T) {
	modes := defaultHostKVSupportModes(3)
	if modes[0] != hostKVSupportLegacy || modes[1] != hostKVSupportRequest || modes[2] != hostKVSupportRequest {
		t.Fatalf("unexpected modes: %#v", modes)
	}
}

func TestParseHostKVSupportModes(t *testing.T) {
	modes, err := parseHostKVSupportModes("legacy,request", 2)
	if err != nil {
		t.Fatalf("parseHostKVSupportModes failed: %v", err)
	}
	if modes[0] != hostKVSupportLegacy || modes[1] != hostKVSupportRequest {
		t.Fatalf("unexpected modes: %#v", modes)
	}
}

func TestLoadConfigDefaultsHostKVSupport(t *testing.T) {
	cfg, err := loadConfig(rawFlags{
		hosts:            strptr("http://127.0.0.1:11438,http://127.0.0.1:11439"),
		hostLabels:       strptr(""),
		hostKVSupport:    strptr(""),
		model:            strptr("m"),
		kvModes:          strptr("f16,tq35"),
		profile:          strptr("quick"),
		workloads:        strptr(""),
		numCtx:           strptr(""),
		concurrency:      strptr(""),
		promptTokens:     intPtr(0),
		maxTokens:        intPtr(0),
		warmup:           intPtr(-1),
		epochs:           intPtr(-1),
		keepAlive:        floatPtr(-1),
		seed:             intPtr(42),
		temperature:      floatPtr(0),
		timeout:          intPtr(60),
		stream:           testBoolPtr(true),
		failFast:         testBoolPtr(false),
		probeIntervalMS:  intPtr(100),
		output:           strptr(""),
		jsonlOutput:      strptr(""),
		summaryOutput:    strptr(""),
		outputDir:        strptr("/results"),
		captureOllamaPS:  testBoolPtr(false),
		captureGPU:       testBoolPtr(false),
		captureRunnerRSS: testBoolPtr(false),
		debug:            testBoolPtr(false),
		progress:         strptr("auto"),
		progressWidth:    intPtr(30),
	})
	if err != nil {
		t.Fatalf("loadConfig failed: %v", err)
	}
	if cfg.Hosts[0].KVSupportMode != hostKVSupportLegacy || cfg.Hosts[1].KVSupportMode != hostKVSupportRequest {
		t.Fatalf("unexpected host kv support defaults: %#v", cfg.Hosts)
	}
}

func TestImpactProfileDefaults(t *testing.T) {
	if got := profileWarmup("impact"); got != 1 {
		t.Fatalf("profileWarmup(impact) = %d, want 1", got)
	}
	if got := profileEpochs("impact"); got != 1 {
		t.Fatalf("profileEpochs(impact) = %d, want 1", got)
	}
	workloads := profileWorkloads("impact")
	if len(workloads) != 2 || workloads[0] != workloadPrefillHeavy || workloads[1] != workloadDecodeGrowth {
		t.Fatalf("unexpected impact workloads: %#v", workloads)
	}
	contexts := profileContexts("impact")
	if len(contexts) != 2 || contexts[0] != 16384 || contexts[1] != 8192 {
		t.Fatalf("unexpected impact contexts: %#v", contexts)
	}
	concurrency := profileConcurrency("impact")
	if len(concurrency) != 2 || concurrency[0] != 2 || concurrency[1] != 1 {
		t.Fatalf("unexpected impact concurrency: %#v", concurrency)
	}
}

func TestProofProfilesDefaults(t *testing.T) {
	if got := defaultKVModes("regression"); len(got) != 1 || got[0] != "f16" {
		t.Fatalf("unexpected regression kv modes: %#v", got)
	}
	if got := defaultKVModes("turbo-benefit"); len(got) != 3 || got[1] != "tq25" || got[2] != "tq35" {
		t.Fatalf("unexpected turbo-benefit kv modes: %#v", got)
	}
	if got := profileEpochs("regression"); got != 8 {
		t.Fatalf("profileEpochs(regression) = %d, want 8", got)
	}
	if got := profileEpochs("capacity"); got != 2 {
		t.Fatalf("profileEpochs(capacity) = %d, want 2", got)
	}
	if got := defaultKVModes("spill"); len(got) != 3 || got[2] != "tq35" {
		t.Fatalf("unexpected spill kv modes: %#v", got)
	}
	if got := profileEpochs("spill"); got != 2 {
		t.Fatalf("profileEpochs(spill) = %d, want 2", got)
	}
}

func TestTimeoutDurationZeroDisablesTimeout(t *testing.T) {
	if got := timeoutDuration(0); got != 0 {
		t.Fatalf("timeoutDuration(0) = %v, want 0", got)
	}
	if got := timeoutDuration(-1); got != 0 {
		t.Fatalf("timeoutDuration(-1) = %v, want 0", got)
	}
	if got := timeoutDuration(60); got.Seconds() != 60 {
		t.Fatalf("timeoutDuration(60) = %v, want 60s", got)
	}
}
