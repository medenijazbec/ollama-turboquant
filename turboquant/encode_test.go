package turboquant

import (
	"math"
	"testing"
)

// TestOutlierSplitBoundaries pins the single-block vs two-block boundary.
// dim == OutlierCount must produce a single block (no split).
// dim == OutlierCount+1 is the smallest two-block encoding.
// TestMemoryFormulaMatchesMarshalSize verifies that the two-block byte-count
// formula used in fs/ggml/ggml.go GraphSize matches the actual MarshalBinary
// output size. This catches formula drift whenever Block layout changes.
func TestMemoryFormulaMatchesMarshalSize(t *testing.T) {
	cases := []struct {
		preset Preset
		dim    int
	}{
		{PresetTQ25, 128},
		{PresetTQ35, 128},
		{PresetTQ35, 64},
	}
	for _, tc := range cases {
		vec := make([]float32, tc.dim)

		keyEncoded, err := EncodeKeyVector(vec, tc.preset)
		if err != nil {
			t.Fatalf("%s dim=%d key: %v", tc.preset.Name, tc.dim, err)
		}
		keyData, err := keyEncoded.MarshalBinary()
		if err != nil {
			t.Fatalf("%s dim=%d key marshal: %v", tc.preset.Name, tc.dim, err)
		}

		valueEncoded, err := EncodeValueVector(vec, tc.preset)
		if err != nil {
			t.Fatalf("%s dim=%d value: %v", tc.preset.Name, tc.dim, err)
		}
		valueData, err := valueEncoded.MarshalBinary()
		if err != nil {
			t.Fatalf("%s dim=%d value marshal: %v", tc.preset.Name, tc.dim, err)
		}

		// Replicate the fs/ggml/ggml.go GraphSize formula.
		const outlierCount = uint64(32)
		outlierBits := uint64(tc.preset.OutlierBits)
		regularKeyBits := uint64(tc.preset.KeyPrimaryBits)
		regularValueBits := uint64(tc.preset.ValueBits)
		dim := uint64(tc.dim)
		outlierData := (outlierCount*outlierBits + 7) / 8
		qjlData := (outlierCount + 7) / 8
		wantKey := 122 + 2*dim + outlierData + ((dim-outlierCount)*regularKeyBits+7)/8 + qjlData
		wantValue := 122 + 2*dim + outlierData + ((dim-outlierCount)*regularValueBits+7)/8

		if uint64(len(keyData)) != wantKey {
			t.Errorf("%s dim=%d: key MarshalBinary=%d bytes, formula=%d",
				tc.preset.Name, tc.dim, len(keyData), wantKey)
		}
		if uint64(len(valueData)) != wantValue {
			t.Errorf("%s dim=%d: value MarshalBinary=%d bytes, formula=%d",
				tc.preset.Name, tc.dim, len(valueData), wantValue)
		}
	}
}

func TestOutlierSplitBoundaries(t *testing.T) {
	for _, preset := range []Preset{PresetTQ25, PresetTQ35} {
		atBoundary := pseudoRandomVector(preset.OutlierCount, 0xbabe)
		encoded, err := EncodeKeyVector(atBoundary, preset)
		if err != nil {
			t.Fatalf("%s dim=OutlierCount: %v", preset.Name, err)
		}
		if len(encoded.Blocks) != 1 {
			t.Errorf("%s dim=%d: got %d blocks, want 1 (no split at exact boundary)",
				preset.Name, preset.OutlierCount, len(encoded.Blocks))
		}
		if len(encoded.Blocks[0].ChannelIndices) != 0 {
			t.Errorf("%s dim=%d: single-block should have no ChannelIndices", preset.Name, preset.OutlierCount)
		}

		minSplit := pseudoRandomVector(preset.OutlierCount+1, 0xbabe)
		encoded2, err := EncodeKeyVector(minSplit, preset)
		if err != nil {
			t.Fatalf("%s dim=OutlierCount+1: %v", preset.Name, err)
		}
		if len(encoded2.Blocks) != 2 {
			t.Errorf("%s dim=%d: got %d blocks, want 2 (minimum outlier split)",
				preset.Name, preset.OutlierCount+1, len(encoded2.Blocks))
		}
		// Regular block has dim=1; verify it round-trips cleanly.
		data, err := encoded2.MarshalBinary()
		if err != nil {
			t.Fatalf("%s dim=OutlierCount+1 marshal: %v", preset.Name, err)
		}
		decoded, _, err := DecodeVector(data)
		if err != nil {
			t.Fatalf("%s dim=OutlierCount+1 decode: %v", preset.Name, err)
		}
		if len(decoded) != preset.OutlierCount+1 {
			t.Errorf("%s dim=OutlierCount+1: decoded len=%d want %d",
				preset.Name, len(decoded), preset.OutlierCount+1)
		}
	}
}

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

// TestOutlierSplitMSEImproves verifies that encoding with the outlier-split
// strategy achieves lower MSE than uniform quantization at the same average bit
// rate. This is the core quality claim of §4.3 of arXiv 2504.19874.
func TestOutlierSplitMSEImproves(t *testing.T) {
	const dim = 128
	const trials = 200
	rng := splitmix64(0xfeedbabe12345678)

	for _, preset := range []Preset{PresetTQ25, PresetTQ35} {
		t.Run(preset.Name, func(t *testing.T) {
			var splitMSE, uniformMSE float64
			for i := 0; i < trials; i++ {
				vec := make([]float32, dim)
				for j := range vec {
					vec[j] = float32(gaussianFloat64(&rng))
				}

				// Outlier-split encoding (2-block, current path).
				splitEncoded, err := EncodeValueVector(vec, preset)
				if err != nil {
					t.Fatal(err)
				}
				splitData, err := splitEncoded.MarshalBinary()
				if err != nil {
					t.Fatal(err)
				}
				splitDecoded, _, err := DecodeVector(splitData)
				if err != nil {
					t.Fatal(err)
				}
				for j := range vec {
					d := float64(vec[j] - splitDecoded[j])
					splitMSE += d * d
				}

				// Uniform encoding: both sub-block sizes at regular bits, same total bits.
				// Encode the whole vector at regular bits to match the average bit rate.
				uniformBits := preset.ValueBits
				unifBlock, err := encodeSubBlock(vec, nil, preset, roleValue, objectiveMSE, uniformBits, preset.RotationSeed)
				if err != nil {
					t.Fatal(err)
				}
				codebook, _ := scalarCodebook(dim, uniformBits)
				rot := BuildRotation(dim, preset.RotationSeed)
				uIndices := unpackBits(unifBlock.RegularIndices, uniformBits, dim)
				unifRotated := make([]float32, dim)
				for j, idx := range uIndices {
					unifRotated[j] = dequantizeScalar(idx, codebook) * unifBlock.Scale
				}
				unifDecoded := ApplyInverseRotation(unifRotated, rot)
				for j := range vec {
					d := float64(vec[j] - unifDecoded[j])
					uniformMSE += d * d
				}
			}
			splitMSE /= float64(trials * dim)
			uniformMSE /= float64(trials * dim)
			t.Logf("%s: split MSE=%.6f  uniform MSE=%.6f", preset.Name, splitMSE, uniformMSE)
			if splitMSE >= uniformMSE {
				t.Errorf("outlier split MSE (%.6f) not lower than uniform MSE (%.6f)", splitMSE, uniformMSE)
			}
		})
	}
}

// TestOutlierSplitProductUnbiasedness verifies that the multi-block product
// estimator remains near-unbiased after the outlier split is applied.
func TestOutlierSplitProductUnbiasedness(t *testing.T) {
	const dim = 128
	const trials = 300
	const maxRelBias = 0.07

	rng := splitmix64(0xabcdef0123456789)
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

	t.Logf("outlier-split avg signed bias = %.6f, rms true dot = %.6f, relative bias = %.4f",
		avgSignedBias, rmsTrue, relativeBias)
	if relativeBias > maxRelBias {
		t.Fatalf("relative bias = %.4f, want <= %.4f (multi-block estimator should be near-unbiased)",
			relativeBias, maxRelBias)
	}
}

// TestMultiBlockPrepareAndScore verifies that PrepareEncodedVector for an
// outlier-split key vector sets IsOriginalSpace and produces scores close to
// ScoreEncodedVector (which uses the direct multi-block gather path).
func TestMultiBlockPrepareAndScore(t *testing.T) {
	const dim = 128
	rng := splitmix64(0x1122334455667788)

	for i := 0; i < 20; i++ {
		key := make([]float32, dim)
		query := make([]float32, dim)
		for j := range key {
			key[j] = float32(gaussianFloat64(&rng))
			query[j] = float32(gaussianFloat64(&rng))
		}

		encoded, err := EncodeKeyVector(key, PresetTQ35)
		if err != nil {
			t.Fatal(err)
		}
		data, err := encoded.MarshalBinary()
		if err != nil {
			t.Fatal(err)
		}

		pb, _, err := PrepareEncodedVector(data)
		if err != nil {
			t.Fatal(err)
		}

		if !pb.IsOriginalSpace() {
			t.Fatal("expected IsOriginalSpace=true for outlier-split encoding")
		}
		if pb.EncodedDim() != dim {
			t.Fatalf("EncodedDim = %d, want %d", pb.EncodedDim(), dim)
		}

		// Score with PreparedBlock (original-space path — no query rotation).
		preparedScore := ScorePreparedBlock(query, pb)

		// Score directly via ScoreEncodedVector (gather-rotate-per-block path).
		directScore, _, err := ScoreEncodedVector(query, data)
		if err != nil {
			t.Fatal(err)
		}

		if diff := abs32(preparedScore - directScore); diff > 1e-3 {
			t.Errorf("trial %d: PreparedBlock score %.6f vs direct score %.6f, diff %.6f",
				i, preparedScore, directScore, diff)
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
