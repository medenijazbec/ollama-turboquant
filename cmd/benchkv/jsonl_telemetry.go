package main

import (
	"encoding/json"
	"os"
	"sync"
)

type telemetryWriter struct {
	mu   sync.Mutex
	f    *os.File
	enc  *json.Encoder
	path string
}

func newTelemetryWriter(path string) (*telemetryWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &telemetryWriter{
		f:    f,
		enc:  json.NewEncoder(f),
		path: path,
	}, nil
}

func (w *telemetryWriter) write(sample liveSample) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.enc == nil {
		return nil
	}
	if err := w.enc.Encode(sample); err != nil {
		return err
	}
	return w.f.Sync()
}

func (w *telemetryWriter) close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.f == nil {
		return nil
	}
	err := w.f.Close()
	w.f = nil
	w.enc = nil
	return err
}
