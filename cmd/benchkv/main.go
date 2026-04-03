package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	flags := parseFlags()
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS]\n\n", os.Args[0])
		fmt.Fprintln(os.Stderr, "KV-cache stress benchmark for native Ollama hosts.")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		fmt.Fprintln(os.Stderr)
		fmt.Fprintf(os.Stderr, "Example:\n")
		fmt.Fprintf(os.Stderr, "  %s --hosts http://127.0.0.1:11438,http://127.0.0.1:11439 --model qwen3-30b:instruct2507-udq4kxl --kv-modes f16,q8_0,q4_0,tq25,tq35 --profile full --output /results/qwen30b_kvstress.csv\n", os.Args[0])
	}
	flag.Parse()

	cfg, err := loadConfig(flags)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	if err := ensureOutputDirs(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	tracker := newProgressTracker(cfg.ProgressMode, cfg.ProgressWidth, estimateTotalUnits(cfg), cfg.Debug)
	workers, aggregates, staircases, err := runBenchmark(cfg, tracker)
	tracker.Finish()
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
	}

	if writeErr := writeJSONL(cfg.JSONLPath, workers); writeErr != nil {
		fmt.Fprintf(os.Stderr, "ERROR writing JSONL: %v\n", writeErr)
		os.Exit(1)
	}
	if writeErr := writeCSV(cfg.OutputPath, aggregates); writeErr != nil {
		fmt.Fprintf(os.Stderr, "ERROR writing CSV: %v\n", writeErr)
		os.Exit(1)
	}
	if writeErr := writeSummary(cfg.MarkdownPath, aggregates, staircases); writeErr != nil {
		fmt.Fprintf(os.Stderr, "ERROR writing summary: %v\n", writeErr)
		os.Exit(1)
	}

	fmt.Printf("Wrote CSV summary to %s\n", cfg.OutputPath)
	fmt.Printf("Wrote JSONL rows to %s\n", cfg.JSONLPath)
	fmt.Printf("Wrote markdown summary to %s\n", cfg.MarkdownPath)

	if err != nil {
		os.Exit(1)
	}
}
