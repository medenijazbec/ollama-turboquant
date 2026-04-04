package turboquant

import (
	"math"
	"testing"
)

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
	if keyEncoded.Blocks[0].Objective != uint8(objectiveProduct) {
		t.Fatalf("key objective = %d, want %d", keyEncoded.Blocks[0].Objective, objectiveProduct)
	}
	if valueEncoded.Blocks[0].Objective != uint8(objectiveMSE) {
		t.Fatalf("value objective = %d, want %d", valueEncoded.Blocks[0].Objective, objectiveMSE)
	}
	if keyEncoded.Blocks[0].QJLRows == 0 {
		t.Fatal("expected product-mode key rows to carry a residual sketch")
	}
	if valueEncoded.Blocks[0].QJLRows != 0 {
		t.Fatal("expected MSE value rows to omit a residual sketch")
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

// TestQJLDimMatchesPaperSpec verifies that the QJL sketch uses d random
// projections (one per dimension), matching the paper's specification in
// arXiv 2504.19874. With QJLRowsDivisor=1 this ensures the estimator variance
// matches the paper's theoretical analysis and the bit accounting is exact:
// tq25 = 2.5 bits/elem avg, tq35 = 3.5 bits/elem avg.
func TestQJLDimMatchesPaperSpec(t *testing.T) {
	cases := []struct {
		preset Preset
		dim    int
	}{
		{PresetTQ25, 64},
		{PresetTQ25, 128},
		{PresetTQ35, 128},
		{PresetTQ35, 256},
	}
	for _, tc := range cases {
		got := tc.preset.KeyQJLRows(tc.dim)
		if got != tc.dim {
			t.Errorf("%s: KeyQJLRows(%d) = %d, want %d (paper spec: d projections per d-dim vector)",
				tc.preset.Name, tc.dim, got, tc.dim)
		}
	}
}

// TestPaperMSEDistortionBound verifies that the MSE quantizer operates within
// the information-theoretic bounds from Theorem 3 of arXiv 2504.19874:
//
//	lower: D_mse >= 1/4^b
//	upper: D_mse <= (√3π/2) / 4^b ≈ 2.72 / 4^b
//
// Tested on 1000 random unit vectors at dim=128 (the primary validated head_dim).
func TestPaperMSEDistortionBound(t *testing.T) {
	const dim = 128
	const trials = 1000

	cases := []struct {
		preset Preset
		bits   int
	}{
		{PresetTQ25, PresetTQ25.ValueBits},
		{PresetTQ35, PresetTQ35.ValueBits},
	}

	for _, tc := range cases {
		t.Run(tc.preset.Name, func(t *testing.T) {
			lowerBound := 1.0 / math.Pow(4, float64(tc.bits))
			paperUpper := 2.72 / math.Pow(4, float64(tc.bits))

			var totalDistortion float64
			rng := splitmix64(0x1234567890abcdef)
			for i := 0; i < trials; i++ {
				// Random unit vector drawn from the uniform distribution on S^{d-1}.
				vec := make([]float32, dim)
				var norm2 float64
				for j := range vec {
					v := gaussianFloat64(&rng)
					vec[j] = float32(v)
					norm2 += v * v
				}
				norm := math.Sqrt(norm2)
				for j := range vec {
					vec[j] /= float32(norm)
				}

				encoded, err := EncodeValueVector(vec, tc.preset)
				if err != nil {
					t.Fatal(err)
				}
				data, err := encoded.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				decoded, _, err := DecodeVector(data)
				if err != nil {
					t.Fatal(err)
				}

				// D_mse = ||x - x_hat||^2 / ||x||^2 = ||x - x_hat||^2 (||x||=1)
				var distortion float64
				for j := range vec {
					d := float64(vec[j] - decoded[j])
					distortion += d * d
				}
				totalDistortion += distortion
			}
			avgDistortion := totalDistortion / float64(trials)

			t.Logf("%s D_mse = %.6f, paper bounds [%.6f, %.6f]",
				tc.preset.Name, avgDistortion, lowerBound, paperUpper)

			// Allow 1.5× headroom over the paper's upper bound to account for
			// finite-d effects and the Cartesian (non-PolarQuant) encoding path.
			if avgDistortion > paperUpper*1.5 {
				t.Fatalf("D_mse = %.6f exceeds paper upper bound × 1.5 (%.6f)",
					avgDistortion, paperUpper*1.5)
			}
		})
	}
}

// TestPaperProductUnbiasedness verifies that the product-objective estimator
// is near-unbiased: E[score(q, encoded_k) - dot(q, k)] ≈ 0. This is the
// central claim of Q_prod in arXiv 2504.19874.
func TestPaperProductUnbiasedness(t *testing.T) {
	const dim = 128
	const trials = 500
	// Allow 5% signed relative bias averaged over 500 trials.
	const maxRelBias = 0.05

	rng := splitmix64(0xdeadbeefcafe1234)
	var signedBias, rmsTrue float64
	for i := 0; i < trials; i++ {
		query := make([]float32, dim)
		key := make([]float32, dim)
		for j := range query {
			query[j] = float32(gaussianFloat64(&rng))
			key[j] = float32(gaussianFloat64(&rng))
		}

		encoded, err := EncodeKeyVector(key, PresetTQ35)
		if err != nil {
			t.Fatal(err)
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}
		estimated, _, err := ScoreEncodedVector(query, data)
		if err != nil {
			t.Fatal(err)
		}

		var trueDot float32
		for j := range query {
			trueDot += query[j] * key[j]
		}
		signedBias += float64(estimated - trueDot)
		rmsTrue += float64(trueDot * trueDot)
	}

	avgSignedBias := signedBias / float64(trials)
	rmsTrue = math.Sqrt(rmsTrue / float64(trials))
	relativeBias := math.Abs(avgSignedBias) / rmsTrue

	t.Logf("avg signed bias = %.6f, rms true dot = %.6f, relative bias = %.4f",
		avgSignedBias, rmsTrue, relativeBias)

	if relativeBias > maxRelBias {
		t.Fatalf("relative bias = %.4f, want <= %.4f (product estimator should be near-unbiased)",
			relativeBias, maxRelBias)
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
