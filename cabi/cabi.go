package main

/*
#include <stdint.h>
*/
import "C"

import (
	"sync"
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

// ─────────────────────────────── utils ───────────────────────────────────────

func boolToC(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// `layers` is [ (w,h), (w,h), ... ] contiguous in memory
func decodeLayers(ptr unsafe.Pointer, n C.int) []struct{ Width, Height int } {
	sz := int(n)
	out := make([]struct{ Width, Height int }, sz)
	// data is int32 pairs
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

// Create a new network (float32) and return a handle.
// layers: int32 array of length 2*L: [w0,h0,w1,h1,...]
// activations: array of char* (L items), e.g. {"linear","relu","softmax"}
// fullyConnected: bytes (0/1) (L items) – input entry is ignored; starting from layer1.
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

	nn := paragon.NewNetwork[float32](layerSizes, activations, fully)

	// Optional: enable GPU fast path (float32 path)
	if enableWebGPU != 0 {
		nn.WebGPUNative = true
		_ = nn.InitializeOptimizedGPU() // if this fails, Paragon falls back to CPU
	}

	return C.longlong(putNet(nn))
}

// Destroy and free a network.
//
//export Paracast_Free
func Paracast_Free(h C.longlong) {
	delNet(handle(h))
}

// Load a model from JSON string; returns handle.
//
//export Paracast_LoadFromJSON
func Paracast_LoadFromJSON(jsonStr *C.char, enableWebGPU C.int) C.longlong {
	if jsonStr == nil {
		return 0
	}
	goStr := C.GoString(jsonStr)

	anyNet, err := paragon.LoadNamedNetworkFromJSONString(goStr)
	if err != nil {
		return 0
	}

	nf, ok := anyNet.(*paragon.Network[float32])
	if !ok {
		// For v3.1.3, require float32 models; return 0 if the JSON encodes another type.
		return 0
	}

	if enableWebGPU != 0 {
		nf.WebGPUNative = true
		_ = nf.InitializeOptimizedGPU() // falls back to CPU automatically if init fails
	}

	return C.longlong(putNet(nf))
}

// Save a model to JSON file path.
//
//export Paracast_SaveJSON
func Paracast_SaveJSON(h C.longlong, path *C.char) C.int {
	n := getNet(handle(h))
	if n == nil {
		return 0
	}
	if err := n.SaveJSON(C.GoString(path)); err != nil {
		return 0
	}
	return 1
}

// Forward pass on a single sample.
// `input` is row-major float32 of size (inH*inW).
//
//export Paracast_Forward
func Paracast_Forward(h C.longlong, input *C.float, inW C.int, inH C.int) C.int {
	n := getNet(handle(h))
	if n == nil {
		return 0
	}

	w, hgt := int(inW), int(inH)
	rowMajor := unsafe.Slice((*float32)(unsafe.Pointer(input)), w*hgt)

	// reshape to [][]float64 for Paragon API (it accepts [][]float64)
	sample := make([][]float64, hgt)
	for y := 0; y < hgt; y++ {
		row := make([]float64, w)
		for x := 0; x < w; x++ {
			row[x] = float64(rowMajor[y*w+x])
		}
		sample[y] = row
	}

	n.Forward(sample) // Paragon forward (GPU/CPU inside)
	n.ApplySoftmax()  // if final act is softmax, this is a no-op otherwise
	return 1
}

// Read the output vector (flattened) into `out` (float32).
//
//export Paracast_GetOutput
func Paracast_GetOutput(h C.longlong, out *C.float, outMax C.int) C.int {
	n := getNet(handle(h))
	if n == nil {
		return 0
	}

	// Paragon helper that returns the last layer values as []float64
	values := n.GetOutput()

	max := int(outMax)
	if len(values) > max {
		return 0
	}
	dst := unsafe.Slice((*float32)(unsafe.Pointer(out)), len(values))
	for i := range values {
		dst[i] = float32(values[i])
	}
	return C.int(len(values))
}

// Train for N epochs on (inputs, targets).
// Inputs/targets are packed batches of samples back-to-back (row-major each).
// dims: inW,inH,outW,outH. batch is the number of samples provided.
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
		return 0
	}

	b := int(batch)
	iw, ih := int(inW), int(inH)
	ow, oh := int(outW), int(outH)

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

	if useGPU != 0 {
		n.WebGPUNative = true
		_ = n.InitializeOptimizedGPU()
	}

	// Minimal training loop: epochs × batch (no shuffling for simplicity)
	for e := 0; e < int(epochs); e++ {
		for i := 0; i < b; i++ {
			n.Forward(inB[i])
			// choose GPU-aware train if available in your API; otherwise:
			n.Backward(outB[i], float64(lr), float32(clipUpper), float32(clipLower))
		}
	}

	return 1
}
