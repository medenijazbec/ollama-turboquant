//go:build windows

package main

func newNVMLPoller() (gpuPoller, bool) {
	return nil, false
}
