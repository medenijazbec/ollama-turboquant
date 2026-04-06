//go:build !windows

package main

import "fmt"

import "github.com/NVIDIA/go-nvml/pkg/nvml"

type nvmlPoller struct {
	devices []nvml.Device
}

func newNVMLPoller() (gpuPoller, bool) {
	if ret := nvml.Init(); ret != nvml.SUCCESS {
		return nil, false
	}

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS || count == 0 {
		nvml.Shutdown()
		return nil, false
	}

	devices := make([]nvml.Device, 0, count)
	for i := 0; i < count; i++ {
		device, devRet := nvml.DeviceGetHandleByIndex(i)
		if devRet != nvml.SUCCESS {
			continue
		}
		devices = append(devices, device)
	}
	if len(devices) == 0 {
		nvml.Shutdown()
		return nil, false
	}
	return &nvmlPoller{devices: devices}, true
}

func (p *nvmlPoller) poll() (gpuSample, bool) {
	var totalUsed uint64
	var totalVisible uint64
	var totalUtil float64
	var samples int
	perGPUUsed := make(map[string]int64)
	perGPUFree := make(map[string]int64)
	perGPUUtil := make(map[string]float64)
	perGPUTemp := make(map[string]float64)
	perGPUPower := make(map[string]float64)

	for i, device := range p.devices {
		memInfo, memRet := device.GetMemoryInfo()
		utilRates, utilRet := device.GetUtilizationRates()
		if memRet != nvml.SUCCESS || utilRet != nvml.SUCCESS {
			continue
		}
		totalUsed += memInfo.Used
		totalVisible += memInfo.Total
		totalUtil += float64(utilRates.Gpu)
		key := fmt.Sprintf("%d", i)
		perGPUUsed[key] = int64(memInfo.Used)
		perGPUFree[key] = int64(memInfo.Free)
		perGPUUtil[key] = float64(utilRates.Gpu)
		if temp, ret := device.GetTemperature(nvml.TEMPERATURE_GPU); ret == nvml.SUCCESS {
			perGPUTemp[key] = float64(temp)
		}
		if power, ret := device.GetPowerUsage(); ret == nvml.SUCCESS {
			perGPUPower[key] = float64(power) / 1000
		}
		samples++
	}
	if samples == 0 {
		return gpuSample{}, false
	}

	return gpuSample{
		peakVRAMBytes:         int64(totalUsed),
		meanUtil:              totalUtil / float64(samples),
		totalVisibleVRAMBytes: int64Ptr(int64(totalVisible)),
		perGPUUsedBytes:       perGPUUsed,
		perGPUFreeBytes:       perGPUFree,
		perGPUUtilPercent:     perGPUUtil,
		perGPUTempC:           perGPUTemp,
		perGPUPowerW:          perGPUPower,
		visibleGPUCount:       samples,
	}, true
}

func (p *nvmlPoller) close() {
	nvml.Shutdown()
}

func (p *nvmlPoller) source() string { return "nvml" }
