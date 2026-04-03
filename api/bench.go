package api

import "context"

type BenchScoreRequest struct {
	Model     string         `json:"model"`
	Prompt    string         `json:"prompt"`
	Target    string         `json:"target"`
	Options   map[string]any `json:"options,omitempty"`
	KeepAlive *Duration      `json:"keep_alive,omitempty"`
}

type BenchScoreResponse struct {
	Model                 string  `json:"model"`
	Prompt                string  `json:"prompt,omitempty"`
	Target                string  `json:"target"`
	Observed              string  `json:"observed,omitempty"`
	TokenCount            int     `json:"token_count"`
	NegativeLogLikelihood float64 `json:"negative_log_likelihood"`
	Perplexity            float64 `json:"perplexity"`

	Metrics
}

func (c *Client) BenchScore(ctx context.Context, req *BenchScoreRequest) (*BenchScoreResponse, error) {
	var resp BenchScoreResponse
	if err := c.do(ctx, "POST", "/api/bench/score", req, &resp); err != nil {
		return nil, err
	}

	return &resp, nil
}
