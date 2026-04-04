package turboquant

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestUnmarshalEncodedVectorRejectsBadHeader(t *testing.T) {
	if _, err := UnmarshalEncodedVector([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected malformed header error")
	}
}

func TestUnmarshalEncodedVectorRejectsWrongVersion(t *testing.T) {
	encoded, err := EncodeVector([]float32{1, 2, 3, 4}, PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 99

	if _, err := UnmarshalEncodedVector(data); err == nil {
		t.Fatal("expected unsupported version error")
	}
}

func TestUnmarshalEncodedVectorRejectsBadPresetID(t *testing.T) {
	encoded, err := EncodeVector([]float32{1, 2, 3, 4}, PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	data[1] = 99

	if _, err := UnmarshalEncodedVector(data); err == nil {
		t.Fatal("expected bad preset id error")
	}
}

func TestUnmarshalEncodedVectorRejectsTruncatedBlockPayload(t *testing.T) {
	encoded, err := EncodeVector(pseudoRandomVector(16, 0x99), PresetTQ25)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}

	truncated := data[:len(data)-1]
	if _, err := UnmarshalEncodedVector(truncated); err == nil {
		t.Fatal("expected truncated block payload error")
	}
}

func TestDecodeVectorRejectsInvalidIndexLengths(t *testing.T) {
	encoded, err := EncodeVector(pseudoRandomVector(16, 0x77), PresetTQ35)
	if err != nil {
		t.Fatal(err)
	}
	data, err := encoded.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	ev, err := UnmarshalEncodedVector(data)
	if err != nil {
		t.Fatal(err)
	}

	ev.Blocks[0].RegularIndices = append(ev.Blocks[0].RegularIndices, 0)
	corrupt, err := marshalTestEncodedVector(ev)
	if err != nil {
		t.Fatal(err)
	}

	if _, _, err := DecodeVector(corrupt); err == nil {
		t.Fatal("expected invalid primary index length error")
	}
}

func TestDecodeVectorPreservesOriginalLength(t *testing.T) {
	values := pseudoRandomVector(130, 0x66)
	encoded, err := EncodeVector(values, PresetTQ35)
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
	if len(decoded) != len(values) {
		t.Fatalf("decoded len = %d, want %d", len(decoded), len(values))
	}
}

func TestScoreEncodedVectorMatchesDecodedDotForMSERows(t *testing.T) {
	values := pseudoRandomVector(32, 0x42)
	query := pseudoRandomVector(32, 0x99)

	encoded, err := EncodeVector(values, PresetTQ35)
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
	got, _, err := ScoreEncodedVector(query, data)
	if err != nil {
		t.Fatal(err)
	}

	var want float32
	for i := range query {
		want += query[i] * decoded[i]
	}
	if diff := abs32(got - want); diff > 1e-4 {
		t.Fatalf("score diff = %v, want <= 1e-4", diff)
	}
}

func TestProductModeBiasImprovesOverBaseDot(t *testing.T) {
	trials := 24
	// Use dim=32 (== OutlierCount) so no outlier split is triggered and block[0]
	// covers the full vector. This keeps the test focused on QJL bias reduction
	// without exercising the multi-block path (which is covered separately).
	const dim = 32
	var baseErr25 float32
	var prodErr25 float32
	var baseErr35 float32
	var prodErr35 float32

	for i := 0; i < trials; i++ {
		values := pseudoRandomVector(dim, uint64(100+i))
		query := pseudoRandomVector(dim, uint64(200+i))

		for _, preset := range []Preset{PresetTQ25, PresetTQ35} {
			encoded, err := EncodeKeyVector(values, preset)
			if err != nil {
				t.Fatal(err)
			}

			rotation := BuildRotation(dim, preset.RotationSeed)
			codebook, _ := scalarCodebook(dim, preset.KeyPrimaryBits)
			queryRot := ApplyRotation(query, rotation)
			decodedBaseRot := make([]float32, dim)
			codes := unpackBits(encoded.Blocks[0].RegularIndices, int(encoded.Blocks[0].RegularBits), dim)
			for j, code := range codes {
				decodedBaseRot[j] = dequantizeScalar(code, codebook)
			}
			baseDot := dotFloat32(queryRot, decodedBaseRot)

			data, err := encoded.MarshalBinary()
			if err != nil {
				t.Fatal(err)
			}
			productDot, _, err := ScoreEncodedVector(query, data)
			if err != nil {
				t.Fatal(err)
			}

			want := dotFloat32(query, values)
			switch preset.Name {
			case "tq25":
				baseErr25 += abs32(baseDot - want)
				prodErr25 += abs32(productDot - want)
			case "tq35":
				baseErr35 += abs32(baseDot - want)
				prodErr35 += abs32(productDot - want)
			}
		}
	}

	if prodErr25 > baseErr25 {
		t.Fatalf("tq25 product error = %v, want <= base error %v", prodErr25, baseErr25)
	}
	if prodErr35 > baseErr35 {
		t.Fatalf("tq35 product error = %v, want <= base error %v", prodErr35, baseErr35)
	}
	if prodErr35 > prodErr25 {
		t.Fatalf("tq35 product error = %v, want <= tq25 product error %v", prodErr35, prodErr25)
	}
}

func marshalTestEncodedVector(ev EncodedVector) ([]byte, error) {
	var buf bytes.Buffer
	for _, field := range []any{ev.Version, ev.Preset.ID, uint32(ev.Dim), uint32(len(ev.Blocks))} {
		if err := binary.Write(&buf, binary.LittleEndian, field); err != nil {
			return nil, err
		}
	}

	for _, block := range ev.Blocks {
		blockData, err := block.MarshalBinary()
		if err != nil {
			return nil, err
		}
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(blockData))); err != nil {
			return nil, err
		}
		buf.Write(blockData)
	}

	return buf.Bytes(), nil
}

func dotFloat32(a, b []float32) float32 {
	var out float32
	for i := range a {
		out += a[i] * b[i]
	}
	return out
}
