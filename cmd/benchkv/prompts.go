package main

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

var promptWordList = []string{
	"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"a", "bright", "sunny", "day", "in", "the", "meadow", "where",
	"flowers", "bloom", "and", "birds", "sing", "their", "morning",
	"songs", "while", "gentle", "breeze", "carries", "sweet", "scent",
	"of", "pine", "trees", "across", "rolling", "hills", "toward",
	"distant", "mountains", "covered", "with", "fresh", "snow",
	"beneath", "clear", "blue", "sky", "children", "play", "near",
	"old", "stone", "bridge", "that", "crosses", "winding", "river",
}

type promptGenerator struct {
	mu    sync.Mutex
	cache map[promptKey]promptCalibration
}

func newPromptGenerator() *promptGenerator {
	return &promptGenerator{
		cache: make(map[promptKey]promptCalibration),
	}
}

func (g *promptGenerator) promptForTarget(ctx context.Context, host hostTarget, model string, targetTokens, numCtx int, timeout time.Duration) (promptCalibration, error) {
	key := promptKey{Host: host.BaseURL, Model: model, Target: targetTokens}

	g.mu.Lock()
	if cached, ok := g.cache[key]; ok {
		g.mu.Unlock()
		return cached, nil
	}
	g.mu.Unlock()

	wordCount := max(1, int(math.Round(float64(targetTokens)/1.3)))
	best := promptCalibration{WordCount: wordCount}
	bestDiff := math.MaxFloat64

	for attempt := 0; attempt < 6; attempt++ {
		prompt := renderPrompt(wordCount, 0)
		actual, err := calibratePromptTokens(ctx, host.Client, model, prompt, numCtx, timeout)
		if err != nil {
			return promptCalibration{}, err
		}

		diff := math.Abs(float64(actual-targetTokens)) / math.Max(1, float64(targetTokens))
		if diff < bestDiff {
			bestDiff = diff
			best = promptCalibration{WordCount: wordCount, ActualTokens: actual}
		}
		if diff <= 0.02 {
			best = promptCalibration{WordCount: wordCount, ActualTokens: actual}
			break
		}

		wordCount = max(1, int(math.Round(float64(wordCount)*float64(targetTokens)/math.Max(1, float64(actual)))))
	}

	g.mu.Lock()
	g.cache[key] = best
	g.mu.Unlock()
	return best, nil
}

func renderPrompt(wordCount, offset int) string {
	words := make([]string, wordCount)
	for i := range words {
		words[i] = promptWordList[(i+offset)%len(promptWordList)]
	}
	return strings.Join(words, " ")
}

func renderWorkerPrompt(cal promptCalibration, epoch, workerIndex int) string {
	offset := (epoch * 17) + (workerIndex * 31)
	return renderPrompt(cal.WordCount, offset)
}

func calibratePromptTokens(ctx context.Context, client *api.Client, model, prompt string, numCtx int, timeout time.Duration) (int, error) {
	stream := true
	keepAlive := api.Duration{Duration: -1}
	req := &api.GenerateRequest{
		Model:     model,
		Prompt:    prompt,
		Raw:       true,
		Stream:    &stream,
		KeepAlive: &keepAlive,
		Options: map[string]any{
			"num_ctx":     numCtx,
			"num_predict": 1,
			"temperature": 0,
			"seed":        42,
		},
	}

	runCtx, cancel := withOptionalTimeout(ctx, timeout)
	defer cancel()

	var promptEvalCount int
	err := client.Generate(runCtx, req, func(resp api.GenerateResponse) error {
		if resp.Done {
			promptEvalCount = resp.PromptEvalCount
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	if promptEvalCount <= 0 {
		return 0, fmt.Errorf("prompt calibration for %s returned no prompt_eval_count", model)
	}
	return promptEvalCount, nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
