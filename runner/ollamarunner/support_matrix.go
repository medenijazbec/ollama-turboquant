package ollamarunner

import (
	"fmt"

	"github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/model"
)

type turboQuantSupportTier string

const (
	turboQuantSupportSafe         turboQuantSupportTier = "safe"
	turboQuantSupportConservative turboQuantSupportTier = "conservative"
	turboQuantSupportExperimental turboQuantSupportTier = "experimental"
	turboQuantSupportUnsupported  turboQuantSupportTier = "unsupported"
)

type turboQuantModelSupport struct {
	HeadDim           int
	ArchitectureClass string
	SupportTier       turboQuantSupportTier
	Reason            string
}

func detectTurboQuantModelSupport(m model.Model) turboQuantModelSupport {
	cfg := m.Backend().Config()
	arch := cfg.Architecture()

	// @AmesianX: multi-stage head_dim detection cascade inspired this resolution path.
	// @fritolays: unsupported head_dim values need explicit fallback, not silent mis-detection.
	headDim := detectHeadDim(cfg)
	if headDim <= 0 {
		return turboQuantModelSupport{
			HeadDim:           0,
			ArchitectureClass: arch,
			SupportTier:       turboQuantSupportUnsupported,
			Reason:            "unable to infer attention head dimension from model config",
		}
	}

	switch {
	case headDim == 128:
		return turboQuantModelSupport{HeadDim: headDim, ArchitectureClass: arch, SupportTier: turboQuantSupportSafe}
	case headDim%32 == 0:
		return turboQuantModelSupport{
			HeadDim:           headDim,
			ArchitectureClass: arch,
			SupportTier:       turboQuantSupportConservative,
			Reason:            "head dimension is aligned but not in the primary validated native grouped-128 lane",
		}
	default:
		return turboQuantModelSupport{
			HeadDim:           headDim,
			ArchitectureClass: arch,
			SupportTier:       turboQuantSupportUnsupported,
			Reason:            fmt.Sprintf("unsupported head_dim=%d for native TurboQuant rollout", headDim),
		}
	}
}

func detectHeadDim(cfg fs.Config) int {
	arch := cfg.Architecture()
	embeddingLength := int(cfg.Uint(arch + ".embedding_length"))
	headCount := int(cfg.Uint(arch + ".attention.head_count"))
	headCountKV := int(cfg.Uint(arch + ".attention.head_count_kv"))
	embedHeadCountK := int(cfg.Uint(arch + ".embedding_head_count_k"))

	switch {
	case embedHeadCountK > 0 && embeddingLength > 0:
		return embeddingLength / embedHeadCountK
	case headCountKV > 0 && embeddingLength > 0:
		return embeddingLength / headCountKV
	case headCount > 0 && embeddingLength > 0:
		return embeddingLength / headCount
	default:
		return 0
	}
}
