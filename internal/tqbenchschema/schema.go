package tqbenchschema

type FitStatus string
type CorruptionStatus string
type CorrectnessStatus string
type ResidencyKind string

const (
	FitStatusFit         FitStatus = "fit"
	FitStatusFallback    FitStatus = "fallback"
	FitStatusFailed      FitStatus = "failed"
	FitStatusUnsupported FitStatus = "unsupported"
	FitStatusSkipped     FitStatus = "skipped"

	CorruptionStatusPass    CorruptionStatus = "PASS"
	CorruptionStatusFail    CorruptionStatus = "FAIL"
	CorruptionStatusSkipped CorruptionStatus = "SKIPPED"

	CorrectnessStatusPass        CorrectnessStatus = "PASS"
	CorrectnessStatusFail        CorrectnessStatus = "FAIL"
	CorrectnessStatusScaffolded  CorrectnessStatus = "SCAFFOLDED"
	CorrectnessStatusSkipped     CorrectnessStatus = "SKIPPED"
	CorrectnessStatusUnsupported CorrectnessStatus = "UNSUPPORTED"

	ResidencyGPUOnly    ResidencyKind = "gpu-vram-only"
	ResidencyMixed      ResidencyKind = "mixed-ram-vram"
	ResidencyCPUAssist  ResidencyKind = "cpu-assisted"
	ResidencyMMapAssist ResidencyKind = "mmap-assisted"
	ResidencyUnknown    ResidencyKind = "unknown"
)
