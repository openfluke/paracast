package main

/*
#include <stdint.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
	"unsafe"

	paragon "github.com/openfluke/paragon/v3"
)

// ────────────────────────────── handle registry ──────────────────────────────
type handle int64

var (
	mu     sync.RWMutex
	nextID handle = 1
	nets          = map[handle]*paragon.Network[float32]{}
)

func putNet(n *paragon.Network[float32]) handle {
	mu.Lock()
	defer mu.Unlock()
	id := nextID
	nextID++
	nets[id] = n
	return id
}

func getNet(h handle) *paragon.Network[float32] {
	mu.RLock()
	defer mu.RUnlock()
	return nets[h]
}

func delNet(h handle) {
	mu.Lock()
	defer mu.Unlock()
	delete(nets, h)
}

// ─────────────────────────────── last error ──────────────────────────────────
var (
	errMu     sync.Mutex
	lastError string
)

func setLastErrorf(format string, a ...any) {
	errMu.Lock()
	lastError = fmt.Sprintf(format, a...)
	errMu.Unlock()
}

//export Paracast_GetLastError
func Paracast_GetLastError() *C.char {
	errMu.Lock()
	defer errMu.Unlock()
	return C.CString(lastError)
}

// ─────────────────────────────── utils ───────────────────────────────────────
func boolToC(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// layers is [(w,h), ...] contiguous int32 pairs
func decodeLayers(ptr unsafe.Pointer, n C.int) []struct{ Width, Height int } {
	sz := int(n)
	out := make([]struct{ Width, Height int }, sz)
	data := unsafe.Slice((*C.int)(ptr), sz*2)
	for i := 0; i < sz; i++ {
		out[i].Width = int(data[i*2+0])
		out[i].Height = int(data[i*2+1])
	}
	return out
}

func decodeStrings(ptr **C.char, n C.int) []string {
	sz := int(n)
	out := make([]string, sz)
	slice := unsafe.Slice(ptr, sz)
	for i := 0; i < sz; i++ {
		out[i] = C.GoString(slice[i])
	}
	return out
}

func decodeBools(ptr *C.uchar, n C.int) []bool {
	sz := int(n)
	bytes := unsafe.Slice((*byte)(unsafe.Pointer(ptr)), sz)
	out := make([]bool, sz)
	for i := range bytes {
		out[i] = bytes[i] != 0
	}
	return out
}

// ─────────────────────────────── C API ───────────────────────────────────────

// Create a new network (float32) and return a handle (0 on failure).
// layers: int32 array of length 2*L: [w0,h0,w1,h1,...]
// activations: array of char* (L items), e.g. {"linear","relu","softmax"}
// fullyConnected: bytes (0/1) (L items) – input entry ignored; starting from layer1.
//
//export Paracast_NewNetwork
func Paracast_NewNetwork(
	layersPtr unsafe.Pointer, layersCount C.int,
	actsPtr **C.char, actsCount C.int,
	fullPtr *C.uchar, fullCount C.int,
	enableWebGPU C.int,
) C.longlong {
	layerSizes := decodeLayers(layersPtr, layersCount)
	activations := decodeStrings(actsPtr, actsCount)
	fully := decodeBools(fullPtr, fullCount)

	nn, err := paragon.NewNetwork[float32](layerSizes, activations, fully)
	if err != nil {
		setLastErrorf("NewNetwork failed: %v", err)
		return 0
	}

	// Optional: try GPU fast path (non-fatal if it fails).
	if enableWebGPU != 0 {
		nn.WebGPUNative = true
		start := time.Now()
		if gpuErr := nn.InitializeOptimizedGPU(); gpuErr != nil {
			setLastErrorf("WebGPU init failed (fallback to CPU): %v (init=%s)", gpuErr, time.Since(start))
			nn.WebGPUNative = false
		}
	}

	return C.longlong(putNet(nn))
}

// Create large network for dramatic GPU vs CPU testing
//
//export Paracast_CreateLargeNetwork
func Paracast_CreateLargeNetwork() C.longlong {
	layerSizes := []struct{ Width, Height int }{
		{28, 28},  // 784 input
		{1024, 1}, // 1024 hidden 1
		{512, 1},  // 512 hidden 2
		{256, 1},  // 256 hidden 3
		{128, 1},  // 128 hidden 4
		{64, 1},   // 64 hidden 5
		{10, 1},   // 10 output
	}
	activations := []string{"linear", "relu", "relu", "relu", "relu", "relu", "softmax"}
	fullyConnected := []bool{true, true, true, true, true, true, true}

	nn, err := paragon.NewNetwork[float32](layerSizes, activations, fullyConnected)
	if err != nil {
		setLastErrorf("CreateLargeNetwork failed: %v", err)
		return 0
	}

	return C.longlong(putNet(nn))
}

// Initialize GPU for a network
//
//export Paracast_InitGPU
func Paracast_InitGPU(h C.longlong) C.int {
	n := getNet(handle(h))
	if n == nil {
		setLastErrorf("InitGPU: invalid handle")
		return 0
	}

	n.WebGPUNative = true
	start := time.Now()
	if err := n.InitializeOptimizedGPU(); err != nil {
		setLastErrorf("InitGPU failed: %v (init=%s)", err, time.Since(start))
		n.WebGPUNative = false
		return 0
	}
	return 1
}

// Disable GPU for a network
//
//export Paracast_DisableGPU
func Paracast_DisableGPU(h C.longlong) {
	n := getNet(handle(h))
	if n != nil {
		if n.WebGPUNative {
			n.CleanupOptimizedGPU()
		}
		n.WebGPUNative = false
	}
}

// Generate random test data
//
//export Paracast_GenerateTestData
func Paracast_GenerateTestData(data *C.float, numSamples C.int, seed C.int) {
	if data == nil {
		setLastErrorf("GenerateTestData: data pointer is nil")
		return
	}

	rand.Seed(int64(seed))
	samples := int(numSamples)

	// Each sample is 28x28 = 784 floats
	dataSlice := unsafe.Slice((*float32)(unsafe.Pointer(data)), samples*784)

	for i := 0; i < samples*784; i++ {
		dataSlice[i] = rand.Float32()
	}
}

// Run forward pass on a batch of samples
//
//export Paracast_ForwardBatch
func Paracast_ForwardBatch(h C.longlong, inputData *C.float, numSamples C.int, outputData *C.float) C.longlong {
	n := getNet(handle(h))
	if n == nil {
		setLastErrorf("ForwardBatch: invalid handle")
		return 0
	}
	if inputData == nil || outputData == nil {
		setLastErrorf("ForwardBatch: input or output data is nil")
		return 0
	}

	samples := int(numSamples)
	inputs := unsafe.Slice((*float32)(unsafe.Pointer(inputData)), samples*784)
	outputs := unsafe.Slice((*float32)(unsafe.Pointer(outputData)), samples*10)

	startTime := time.Now()

	for i := 0; i < samples; i++ {
		// Convert flat input to 28x28 grid
		sample := make([][]float64, 28)
		for y := 0; y < 28; y++ {
			row := make([]float64, 28)
			for x := 0; x < 28; x++ {
				row[x] = float64(inputs[i*784+y*28+x])
			}
			sample[y] = row
		}

		// Forward pass
		n.Forward(sample)
		result := n.GetOutput()

		// Copy output
		for j := 0; j < 10 && j < len(result); j++ {
			outputs[i*10+j] = float32(result[j])
		}
	}

	elapsed := time.Since(startTime)
	return C.longlong(elapsed.Nanoseconds())
}

// Warm up GPU (run several forward passes)
//
//export Paracast_WarmupGPU
func Paracast_WarmupGPU(h C.longlong, warmupSamples C.int) C.longlong {
	n := getNet(handle(h))
	if n == nil {
		setLastErrorf("WarmupGPU: invalid handle")
		return 0
	}

	// Generate a single sample for warmup
	rand.Seed(42)
	sample := make([][]float64, 28)
	for y := 0; y < 28; y++ {
		row := make([]float64, 28)
		for x := 0; x < 28; x++ {
			row[x] = rand.Float64()
		}
		sample[y] = row
	}

	warmups := int(warmupSamples)
	startTime := time.Now()

	for i := 0; i < warmups; i++ {
		n.Forward(sample)
	}

	elapsed := time.Since(startTime)
	return C.longlong(elapsed.Nanoseconds())
}

// Compare two output arrays and return max difference
//
//export Paracast_CompareOutputs
func Paracast_CompareOutputs(output1 *C.float, output2 *C.float, numElements C.int) C.float {
	if output1 == nil || output2 == nil {
		setLastErrorf("CompareOutputs: one or both output arrays is nil")
		return -1.0
	}

	elements := int(numElements)
	arr1 := unsafe.Slice((*float32)(unsafe.Pointer(output1)), elements)
	arr2 := unsafe.Slice((*float32)(unsafe.Pointer(output2)), elements)

	maxDiff := float32(0.0)
	for i := 0; i < elements; i++ {
		diff := arr1[i] - arr2[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	return C.float(maxDiff)
}

// Load a model from JSON string; returns handle (0 on failure).
//
//export Paracast_LoadFromJSON
func Paracast_LoadFromJSON(jsonStr *C.char, enableWebGPU C.int) C.longlong {
	if jsonStr == nil {
		setLastErrorf("LoadFromJSON: input string is nil")
		return 0
	}
	raw := C.GoString(jsonStr)

	// 1) Bootstrap with a trivial float32 model so wgsl scalar type is set.
	//    NewNetwork[float32] sets the internal WGSL type used by shader gen.
	boot, err := paragon.NewNetwork[float32](
		[]struct{ Width, Height int }{{1, 1}}, // 1×1 stub layer
		[]string{"linear"},
		[]bool{true},
	)
	if err != nil {
		setLastErrorf("LoadFromJSON: bootstrap NewNetwork failed: %v", err)
		return 0
	}
	boot.TypeName = "float32"

	// 2) Overwrite the layers/weights from JSON (preserves GPU type metadata).
	if err := boot.UnmarshalJSONModel([]byte(raw)); err != nil {
		setLastErrorf("LoadFromJSON: UnmarshalJSONModel failed: %v", err)
		return 0
	}
	// Guard older JSONs that might omit Type
	if boot.TypeName == "" {
		boot.TypeName = "float32"
	}

	// 3) Optional GPU init (now safe; WGSL type is already configured).
	if enableWebGPU != 0 {
		boot.WebGPUNative = true
		start := time.Now()
		if gpuErr := boot.InitializeOptimizedGPU(); gpuErr != nil {
			setLastErrorf("LoadFromJSON: WebGPU init failed (fallback to CPU): %v (init=%s)", gpuErr, time.Since(start))
			boot.WebGPUNative = false
		}
	}

	return C.longlong(putNet(boot))
}

// Save a model to JSON file path (returns 1 on success, 0 on failure).
//
//export Paracast_SaveJSON
func Paracast_SaveJSON(h C.longlong, path *C.char) C.int {
	n := getNet(handle(h))
	if n == nil {
		setLastErrorf("SaveJSON: invalid handle")
		return 0
	}
	if path == nil {
		setLastErrorf("SaveJSON: path is nil")
		return 0
	}
	if err := n.SaveJSON(C.GoString(path)); err != nil {
		setLastErrorf("SaveJSON failed: %v", err)
		return 0
	}
	return 1
}

// Forward pass on a single sample (returns 1 on success, 0 on failure).
// `input` is row-major float32 of size (inH*inW).
//
//export Paracast_Forward
func Paracast_Forward(h C.longlong, input *C.float, inW C.int, inH C.int) C.int {
	n := getNet(handle(h))
	if n == nil {
		setLastErrorf("Forward: invalid handle")
		return 0
	}
	if input == nil {
		setLastErrorf("Forward: input is nil")
		return 0
	}

	w, hgt := int(inW), int(inH)
	if w <= 0 || hgt <= 0 {
		setLastErrorf("Forward: invalid input dims (%d,%d)", w, hgt)
		return 0
	}
	rowMajor := unsafe.Slice((*float32)(unsafe.Pointer(input)), w*hgt)

	// reshape to [][]float64 for Paragon API
	sample := make([][]float64, hgt)
	for y := 0; y < hgt; y++ {
		row := make([]float64, w)
		for x := 0; x < w; x++ {
			row[x] = float64(rowMajor[y*w+x])
		}
		sample[y] = row
	}

	n.Forward(sample) // GPU/CPU chosen internally
	return 1
}

// Read the output vector (flattened) into `out` (float32).
// Returns number of floats written (>0) or 0 on failure.
//
//export Paracast_GetOutput
func Paracast_GetOutput(h C.longlong, out *C.float, outMax C.int) C.int {
	n := getNet(handle(h))
	if n == nil {
		setLastErrorf("GetOutput: invalid handle")
		return 0
	}
	if out == nil {
		setLastErrorf("GetOutput: out buffer is nil")
		return 0
	}
	values := n.GetOutput()
	if len(values) == 0 {
		setLastErrorf("GetOutput: empty output")
		return 0
	}
	max := int(outMax)
	if len(values) > max {
		setLastErrorf("GetOutput: buffer too small (need %d, have %d)", len(values), max)
		return 0
	}
	dst := unsafe.Slice((*float32)(unsafe.Pointer(out)), len(values))
	for i := range values {
		dst[i] = float32(values[i])
	}
	return C.int(len(values))
}

// Train for N epochs on (inputs, targets).
// Returns 1 on success, 0 on failure.
// Inputs/targets are packed batches of samples back-to-back (row-major each).
// dims: inW,inH,outW,outH. batch is number of samples provided.
// lr is learning rate. If clipUpper<clipLower you get no clipping.
//
//export Paracast_Train
func Paracast_Train(
	h C.longlong,
	inputs *C.float, inW, inH C.int,
	targets *C.float, outW, outH C.int,
	batch C.int,
	epochs C.int,
	lr C.float,
	useGPU C.int,
	clipUpper C.float,
	clipLower C.float,
) C.int {
	n := getNet(handle(h))
	if n == nil {
		setLastErrorf("Train: invalid handle")
		return 0
	}
	if inputs == nil || targets == nil {
		setLastErrorf("Train: inputs or targets is nil")
		return 0
	}

	b := int(batch)
	iw, ih := int(inW), int(inH)
	ow, oh := int(outW), int(outH)
	if b <= 0 || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 {
		setLastErrorf("Train: invalid dims or batch")
		return 0
	}

	inStride := iw * ih
	outStride := ow * oh
	inAll := unsafe.Slice((*float32)(unsafe.Pointer(inputs)), b*inStride)
	outAll := unsafe.Slice((*float32)(unsafe.Pointer(targets)), b*outStride)

	// reshape to Paragon [][][]
	makeGrid := func(flat []float32, stride, w, h, count int) [][][]float64 {
		out := make([][][]float64, count)
		for s := 0; s < count; s++ {
			grid := make([][]float64, h)
			base := s * stride
			for y := 0; y < h; y++ {
				row := make([]float64, w)
				for x := 0; x < w; x++ {
					row[x] = float64(flat[base+y*w+x])
				}
				grid[y] = row
			}
			out[s] = grid
		}
		return out
	}

	inB := makeGrid(inAll, inStride, iw, ih, b)
	outB := makeGrid(outAll, outStride, ow, oh, b)

	// optional GPU attempt (non-fatal)
	if useGPU != 0 {
		n.WebGPUNative = true
		start := time.Now()
		if gpuErr := n.InitializeOptimizedGPU(); gpuErr != nil {
			setLastErrorf("Train: WebGPU init failed (fallback to CPU): %v (init=%s)", gpuErr, time.Since(start))
			n.WebGPUNative = false
		}
	}

	// Minimal training loop: epochs × batch (no shuffling here)
	for e := 0; e < int(epochs); e++ {
		for i := 0; i < b; i++ {
			n.Forward(inB[i])
			// Use the Paragon API's backward/update that supports GPU internally.
			n.Backward(outB[i], float64(lr), float32(clipUpper), float32(clipLower))
		}
	}

	return 1
}

// Destroy and free a network (also cleans up GPU if active).
//
//export Paracast_Free
func Paracast_Free(h C.longlong) {
	n := getNet(handle(h))
	if n != nil {
		// Best-effort GPU cleanup; ignore error on teardown.
		if n.WebGPUNative {
			n.CleanupOptimizedGPU()
			n.WebGPUNative = false
		}
	}
	delNet(handle(h))
}
