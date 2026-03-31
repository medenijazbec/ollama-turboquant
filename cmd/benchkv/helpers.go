package main

func boolPtr(v bool) *bool {
	return &v
}

func int64Ptr(v int64) *int64 {
	return &v
}

func float64Ptr(v float64) *float64 {
	return &v
}

func ptrInt64Value(v *int64) int64 {
	if v == nil {
		return -1
	}
	return *v
}

func ptrFloat64Value(v *float64) float64 {
	if v == nil {
		return -1
	}
	return *v
}
