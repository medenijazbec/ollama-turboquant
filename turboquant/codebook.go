package turboquant

import (
	"math"
	"slices"
	"sync"
)

type codebookCacheKey struct {
	dim  int
	bits int
}

type scalarCodebookCacheValue struct {
	codebook   []float32
	boundaries []float32
}

var scalarCodebookCache sync.Map

func scalarCodebook(dim int, bits int) ([]float32, []float32) {
	key := codebookCacheKey{dim: dim, bits: bits}
	if cached, ok := scalarCodebookCache.Load(key); ok {
		value := cached.(scalarCodebookCacheValue)
		return append([]float32(nil), value.codebook...), append([]float32(nil), value.boundaries...)
	}

	codebook := buildLloydMaxCodebook(dim, bits)
	value := scalarCodebookCacheValue{
		codebook:   codebook,
		boundaries: codebookBoundaries(codebook),
	}
	actual, _ := scalarCodebookCache.LoadOrStore(key, value)
	cached := actual.(scalarCodebookCacheValue)
	return append([]float32(nil), cached.codebook...), append([]float32(nil), cached.boundaries...)
}

func buildLloydMaxCodebook(dim int, bits int) []float32 {
	levels := 1 << bits
	if levels <= 1 {
		return []float32{0}
	}

	samples := standardNormalSamples(dim, bits, 8192)
	slices.Sort(samples)

	centroids := make([]float64, levels)
	for level := 0; level < levels; level++ {
		begin := level * len(samples) / levels
		end := (level + 1) * len(samples) / levels
		if end <= begin {
			end = begin + 1
		}
		centroids[level] = meanFloat64(samples[begin:end])
	}

	for iter := 0; iter < 48; iter++ {
		slices.Sort(centroids)
		bounds := make([]float64, levels-1)
		for i := range bounds {
			bounds[i] = (centroids[i] + centroids[i+1]) / 2
		}

		sums := make([]float64, levels)
		counts := make([]int, levels)
		for _, sample := range samples {
			idx := quantizeScalarFloat64(sample, bounds)
			sums[idx] += sample
			counts[idx]++
		}

		maxDelta := 0.0
		for i := range centroids {
			next := centroids[i]
			if counts[i] > 0 {
				next = sums[i] / float64(counts[i])
			} else if i == 0 {
				next = bounds[0] - 0.25
			} else if i == len(centroids)-1 {
				next = bounds[len(bounds)-1] + 0.25
			} else {
				next = (bounds[i-1] + bounds[i]) / 2
			}
			maxDelta = math.Max(maxDelta, math.Abs(next-centroids[i]))
			centroids[i] = next
		}
		if maxDelta < 1e-6 {
			break
		}
	}

	slices.Sort(centroids)
	codebook := make([]float32, len(centroids))
	for i := range centroids {
		codebook[i] = float32(centroids[i])
	}
	return codebook
}

func standardNormalSamples(dim int, bits int, count int) []float64 {
	rng := splitmix64(uint64(bits+1)<<48 ^ uint64(dim+1)<<16 ^ 0x4d595df4d0f33173)
	out := make([]float64, count)
	for i := range out {
		out[i] = gaussianFloat64(&rng)
	}
	return out
}

func meanFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func codebookBoundaries(codebook []float32) []float32 {
	if len(codebook) < 2 {
		return nil
	}

	out := make([]float32, len(codebook)-1)
	for i := range out {
		out[i] = (codebook[i] + codebook[i+1]) / 2
	}
	return out
}

// explicitCodebook is retained as a compatibility shim for local audit tooling.
// The paper path uses deterministic Lloyd-Max scalar codebooks rather than the
// previous hand-shaped exponent tables.
func explicitCodebook(bits int, _ float64) []float32 {
	codebook, _ := scalarCodebook(0, bits)
	return codebook
}

func quantizeScalarByBoundary(v float32, codebook []float32, boundaries []float32) uint8 {
	if len(codebook) == 0 {
		return 0
	}
	if len(boundaries) != len(codebook)-1 {
		return quantizeScalarNearest(v, codebook)
	}

	idx := 0
	for idx < len(boundaries) && v >= boundaries[idx] {
		idx++
	}
	return uint8(idx)
}

func quantizeScalarFloat64(v float64, boundaries []float64) int {
	idx := 0
	for idx < len(boundaries) && v >= boundaries[idx] {
		idx++
	}
	return idx
}

func quantizeScalarNearest(v float32, codebook []float32) uint8 {
	best := 0
	bestDist := float32(math.MaxFloat32)
	for i, centroid := range codebook {
		d := abs32(v - centroid)
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return uint8(best)
}

func dequantizeScalar(idx uint8, codebook []float32) float32 {
	if int(idx) >= len(codebook) {
		return 0
	}
	return codebook[idx]
}

func regularCodebook(preset Preset) []float32 {
	return preset.RegularCodebook
}

func regularBoundaries(preset Preset) []float32 {
	return preset.RegularBoundaries
}

func outlierCodebook(preset Preset) []float32 {
	return preset.OutlierCodebook
}

func outlierBoundaries(preset Preset) []float32 {
	return preset.OutlierBoundaries
}

func selectOutliers(values []float32, k int) []int {
	type score struct {
		idx int
		val float32
	}
	if k <= 0 {
		return nil
	}
	scores := make([]score, len(values))
	for i, value := range values {
		scores[i] = score{idx: i, val: abs32(value)}
	}
	slices.SortFunc(scores, func(a, b score) int {
		switch {
		case a.val > b.val:
			return -1
		case a.val < b.val:
			return 1
		default:
			return a.idx - b.idx
		}
	})
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = scores[i].idx
	}
	slices.Sort(out)
	return out
}
