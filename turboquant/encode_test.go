package turboquant

import "testing"

func TestEncodeDecodeRoundTripAcrossShapes(t *testing.T) {
	testCases := []struct {
		name   string
		values []float32
		preset Preset
		maxMSE float32
	}{
		{name: "small", values: []float32{0.25, -1.5, 3.25, 0.75, -0.5, 2.0, 1.0}, preset: PresetTQ35, maxMSE: 1.5},
		{name: "non-power-of-two", values: pseudoRandomVector(70, 2), preset: PresetTQ35, maxMSE: 3.0},
		{name: "multi-head-like", values: pseudoRandomVector(128, 3), preset: PresetTQ25, maxMSE: 5.0},
		{name: "constant", values: filledVector(31, 1.5), preset: PresetTQ25, maxMSE: 1.0},
		{name: "zero", values: filledVector(64, 0), preset: PresetTQ35, maxMSE: 0.01},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			encoded, err := EncodeVector(tc.values, tc.preset)
			if err != nil {
				t.Fatal(err)
			}
			data, err := encoded.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			decoded, preset, err := DecodeVector(data)
			if err != nil {
				t.Fatal(err)
			}
			if preset.Name != tc.preset.Name {
				t.Fatalf("preset = %q, want %q", preset.Name, tc.preset.Name)
			}
			if len(decoded) != len(tc.values) {
				t.Fatalf("decoded len = %d, want %d", len(decoded), len(tc.values))
			}

			stats := Compare(tc.values, decoded)
			if stats.MSE > tc.maxMSE {
				t.Fatalf("MSE = %v, want <= %v", stats.MSE, tc.maxMSE)
			}
		})
	}
}

func TestEncodeVectorDeterministicBytes(t *testing.T) {
	values := pseudoRandomVector(96, 0x4242)

	encodedA, err := EncodeVector(values, PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	encodedB, err := EncodeVector(values, PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}

	dataA, err := encodedA.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	dataB, err := encodedB.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	if string(dataA) != string(dataB) {
		t.Fatal("expected byte-identical encoding output")
	}
}

func TestEncodeKeyAndValueUseDifferentObjectives(t *testing.T) {
	values := pseudoRandomVector(32, 0x77)
	keyEncoded, err := EncodeKeyVector(values, PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	valueEncoded, err := EncodeValueVector(values, PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	if keyEncoded.Blocks[0].Objective != uint8(objectiveMSE) {
		t.Fatalf("key objective = %d, want %d", keyEncoded.Blocks[0].Objective, objectiveMSE)
	}
	if valueEncoded.Blocks[0].Objective != uint8(objectiveMSE) {
		t.Fatalf("value objective = %d, want %d", valueEncoded.Blocks[0].Objective, objectiveMSE)
	}
	if keyEncoded.Blocks[0].QJLRows != 0 {
		t.Fatal("expected stable default key rows to omit a residual sketch")
	}
	if valueEncoded.Blocks[0].QJLRows != 0 {
		t.Fatal("expected stable default value rows to omit a residual sketch")
	}
}

func TestEncodeKeyVectorExperimentalQJL(t *testing.T) {
	values := pseudoRandomVector(32, 0x79)
	keyEncoded, err := EncodeKeyVectorWithOptions(values, PresetTQ35, EncodeOptions{EnableQJLK: true})
	if err != nil {
		t.Fatal(err)
	}
	if keyEncoded.Blocks[0].Objective != uint8(objectiveProduct) {
		t.Fatalf("key objective = %d, want %d", keyEncoded.Blocks[0].Objective, objectiveProduct)
	}
	if keyEncoded.Blocks[0].QJLRows == 0 {
		t.Fatal("expected experimental K-side QJL rows to carry a residual sketch")
	}
}

func TestPresetAliases(t *testing.T) {
	for _, name := range []string{"tq25", "tq35"} {
		preset, err := PresetByName(name)
		if err != nil {
			t.Fatal(err)
		}
		if preset.Name != name {
			t.Fatalf("preset %q resolved to %q", name, preset.Name)
		}
	}

	for _, name := range []string{"tq3", "tq4"} {
		preset, err := PresetByName(name)
		if err != nil {
			t.Fatal(err)
		}
		if preset.Name != "tq35" {
			t.Fatalf("preset %q resolved to %q, want tq35", name, preset.Name)
		}
	}
}

func TestDistortionThresholds(t *testing.T) {
	tq25Mean, err := meanMSEForPreset(PresetTQ25)
	if err != nil {
		t.Fatal(err)
	}
	tq35Mean, err := meanMSEForPreset(PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}

	if tq25Mean > 5.5 {
		t.Fatalf("tq25 mean MSE = %v, want <= 5.5", tq25Mean)
	}
	if tq35Mean > 3.5 {
		t.Fatalf("tq35 mean MSE = %v, want <= 3.5", tq35Mean)
	}
	if tq35Mean > tq25Mean {
		t.Fatalf("tq35 mean MSE = %v, want <= tq25 mean MSE %v", tq35Mean, tq25Mean)
	}
}
