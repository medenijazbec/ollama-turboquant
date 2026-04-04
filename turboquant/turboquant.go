package turboquant

import (
	"fmt"
	"strings"
)

const (
	BlockVersion = 4

	AlgorithmPaper = "paper"
)

type vectorRole uint8

const (
	roleGeneric vectorRole = iota
	roleKey
	roleValue
)

type vectorObjective uint8

const (
	objectiveMSE vectorObjective = iota + 1
	objectiveProduct
)

type Preset struct {
	ID              uint8
	Name            string
	DefaultBlockDim int
	RotationSeed    uint64
	KeyPrimaryBits  int
	ValueBits       int
	QJLRowsDivisor  int
}

var (
	// tq25 uses a 2-bit scalar Lloyd-Max key/value quantizer, with keys gaining
	// a 1-bit QJL residual correction for attention scoring.
	// Keys: 2-bit primary + 1-bit QJL = 3 bits/elem. Values: 2 bits/elem.
	// Average: 2.5 bits/elem -- matches paper's Q_prod at b=3, values at b=2.
	PresetTQ25 = newPreset(1, "tq25", 2, 2, 1, 0x25c0ffee)

	// tq35 uses a 3-bit scalar Lloyd-Max key/value quantizer, with keys gaining
	// a 1-bit QJL residual correction for attention scoring.
	// Keys: 3-bit primary + 1-bit QJL = 4 bits/elem. Values: 3 bits/elem.
	// Average: 3.5 bits/elem -- matches paper's Q_prod at b=4, values at b=3.
	PresetTQ35 = newPreset(2, "tq35", 3, 3, 1, 0x35c0ffee)
)

func newPreset(id uint8, name string, keyBits int, valueBits int, qjlRowsDivisor int, seed uint64) Preset {
	return Preset{
		ID:             id,
		Name:           name,
		DefaultBlockDim: 0,
		RotationSeed:   seed,
		KeyPrimaryBits: keyBits,
		ValueBits:      valueBits,
		QJLRowsDivisor: qjlRowsDivisor,
	}
}

func NormalizePresetName(name string) string {
	switch strings.ToLower(name) {
	case "tq3", "tq4":
		return "tq35"
	default:
		return strings.ToLower(name)
	}
}

func PresetByName(name string) (Preset, error) {
	switch NormalizePresetName(name) {
	case "tq25":
		return PresetTQ25, nil
	case "tq35":
		return PresetTQ35, nil
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset %q", name)
	}
}

func PresetByID(id uint8) (Preset, error) {
	switch id {
	case PresetTQ25.ID:
		return PresetTQ25, nil
	case PresetTQ35.ID:
		return PresetTQ35, nil
	default:
		return Preset{}, fmt.Errorf("unknown turboquant preset id %d", id)
	}
}

func (p Preset) KeyQJLRows(dim int) int {
	if dim <= 0 {
		return 0
	}
	if p.QJLRowsDivisor <= 0 {
		return 0
	}
	rows := dim / p.QJLRowsDivisor
	if rows < 1 {
		return 1
	}
	return rows
}

func blockDimFor(_ Preset, dim int) int {
	return dim
}

func objectiveName(objective vectorObjective) string {
	switch objective {
	case objectiveMSE:
		return "mse"
	case objectiveProduct:
		return "prod"
	default:
		return "unknown"
	}
}
