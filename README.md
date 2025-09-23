# Paracast - C ABI Bindings for Paragon AI

Paracast provides C ABI bindings for the Paragon AI framework, enabling GPU-accelerated neural network inference from C, C++, and any runtime that supports native libraries (Godot, Unity, Unreal, etc.).

## Features

- **GPU Acceleration**: WebGPU-based GPU acceleration with automatic CPU fallback
- **Network Creation**: Build networks from layer specifications or load from JSON
- **Batch Processing**: Efficient batch inference with timing metrics
- **Performance Benchmarking**: Built-in CPU vs GPU performance comparison tools
- **Cross-Platform**: Linux shared libraries (.so) with auto-generated C headers
- **Error Handling**: Comprehensive error reporting and validation

## Quick Start

### Prerequisites

```bash
# Fedora/RHEL
sudo dnf install gcc glibc-devel kernel-headers go

# Ubuntu/Debian  
sudo apt install build-essential golang

# Arch Linux
sudo pacman -S gcc go
```

### Build Everything

```bash
git clone <your-repo>
cd paracast_test
chmod +x build.sh
./build.sh
```

This creates:
- `dist/libparacast.so` - Shared library
- `dist/libparacast.h` - C header file  
- `dist/test_gpu_cpu` - GPU vs CPU benchmark
- `dist/model_benchmark` - JSON model benchmarking tool

## Project Structure

```
paracast_test/
├── README.md              # This file
├── go.mod                 # Go module definition
├── build.sh               # Build script
├── cabi/
│   ├── main.go            # Required empty main() for c-shared
│   └── cabi.go            # C ABI exports
└── dist/                  # Build outputs
    ├── libparacast.so     # Shared library
    ├── libparacast.h      # Auto-generated header
    ├── test_gpu_cpu.c     # GPU benchmark source
    ├── test_gpu_cpu       # GPU benchmark executable  
    ├── model_benchmark.c  # Model benchmark source
    └── model_benchmark    # Model benchmark executable
```

## Usage

### 1. Built-in Network Benchmark

Test GPU acceleration with a preset deep network:

```bash
cd dist
./test_gpu_cpu
```

Expected output:
```
Creating large network for GPU vs CPU comparison...
=== CPU Test ===
CPU Time: 3.309 seconds
CPU Throughput: 302.23 samples/sec

=== GPU Test ===
🚀 GPU Selected: 0x25a2 (0x10de) - Type: discrete-gpu
GPU Time: 0.627 seconds  
GPU Throughput: 1594.71 samples/sec

🚀 GPU DOMINATES! 5.28x faster than CPU!
```

### 2. JSON Model Benchmarking

Benchmark any saved JSON model:

```bash
cd dist

# Basic benchmark
./model_benchmark --model /path/to/your/model.json

# Extended benchmark with more samples and runs
./model_benchmark --model /path/to/model.json --samples 1000 --runs 5

# CPU-only benchmark
./model_benchmark --model /path/to/model.json --cpu-only --runs 10

# GPU-only benchmark
./model_benchmark --model /path/to/model.json --gpu-only --samples 2000
```

### 3. Custom C Integration

```c
#include "libparacast.h"

int main() {
    // Create a network
    long long net = Paracast_CreateLargeNetwork();
    
    // Generate test data (1000 samples of 28x28 pixels)
    float* testData = malloc(1000 * 784 * sizeof(float));
    float* outputs = malloc(1000 * 10 * sizeof(float));
    Paracast_GenerateTestData(testData, 1000, 42);
    
    // CPU inference
    Paracast_DisableGPU(net);
    long long cpuTime = Paracast_ForwardBatch(net, testData, 1000, outputs);
    
    // GPU inference  
    Paracast_InitGPU(net);
    long long gpuTime = Paracast_ForwardBatch(net, testData, 1000, outputs);
    
    printf("CPU: %lld ns, GPU: %lld ns\n", cpuTime, gpuTime);
    printf("GPU speedup: %.2fx\n", (double)cpuTime / gpuTime);
    
    Paracast_Free(net);
    free(testData);
    free(outputs);
    return 0;
}
```

Compile with:
```bash
gcc your_program.c -I./dist -L./dist -lparacast -Wl,-rpath,'$ORIGIN' -o your_program
```

```bash
gcc dist/model_benchmark.c -I./dist -L./dist -lparacast -Wl,-rpath,'$ORIGIN' -o dist/model_benchmark
```


```bash
gcc dist/test_gpu_cpu.c -I./dist -L./dist -lparacast -Wl,-rpath,'$ORIGIN' -o dist/test_gpu_cpu
```

# Basic benchmark
./model_benchmark --model mnist_model.json

# Extended benchmark
./model_benchmark --model mnist_model.json --samples 1000 --runs 5

# CPU or GPU only
./model_benchmark --model mnist_model.json --cpu-only
./model_benchmark --model mnist_model.json --gpu-only

## API Reference

### Network Management
- `Paracast_CreateLargeNetwork()` - Create preset deep network (6 layers)
- `Paracast_NewNetwork(...)` - Create custom network from specifications  
- `Paracast_LoadFromJSON(json, gpu)` - Load network from JSON string
- `Paracast_SaveJSON(handle, path)` - Save network to JSON file
- `Paracast_Free(handle)` - Free network and GPU resources

### GPU Control
- `Paracast_InitGPU(handle)` - Initialize GPU acceleration
- `Paracast_DisableGPU(handle)` - Disable GPU, use CPU
- `Paracast_WarmupGPU(handle, samples)` - GPU warmup with timing

### Inference
- `Paracast_Forward(handle, input, w, h)` - Single sample inference
- `Paracast_ForwardBatch(handle, inputs, count, outputs)` - Batch inference with timing
- `Paracast_GetOutput(handle, output, maxSize)` - Get network output

### Utilities
- `Paracast_GenerateTestData(data, samples, seed)` - Generate random test data
- `Paracast_CompareOutputs(out1, out2, count)` - Compare output arrays
- `Paracast_GetLastError()` - Get last error message
- `Paracast_Train(...)` - Training with GPU support

## Build Script Details

The `build.sh` script:

1. **Creates directory structure**
2. **Builds Go shared library**: Uses `go build -buildmode=c-shared` 
3. **Generates C header**: Auto-created from Go export comments
4. **Compiles test programs**: Links against the shared library
5. **Sets up runtime paths**: Uses `$ORIGIN` for portable execution

Key build flags:
- `-buildmode=c-shared` - Creates shared library + header
- `-Wl,-rpath,'$ORIGIN'` - Library search in executable directory
- `-O3` - Optimization for C programs

## Performance Expectations

Typical results on discrete GPU hardware:

| Network Size | CPU (samples/sec) | GPU (samples/sec) | Speedup |
|--------------|-------------------|-------------------|---------|
| Small (3 layers) | 4,500 | 6,200 | 1.4x |
| Medium (4 layers) | 1,200 | 4,800 | 4.0x |  
| Large (6+ layers) | 300 | 1,600 | 5.3x |

Performance scales with:
- **Network depth** (more layers = better GPU utilization)
- **Layer width** (larger matrices = more parallelism)
- **Batch size** (GPU prefers larger batches)
- **GPU memory bandwidth** (faster memory = better performance)

## Troubleshooting

### Build Issues

**Missing Go compiler:**
```bash
# Install Go 1.21+
wget https://go.dev/dl/go1.21.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
```

**CGO disabled:**
```bash
export CGO_ENABLED=1
go build -buildmode=c-shared -o dist/libparacast.so ./cabi
```

**Missing GCC:**
```bash
sudo dnf install gcc glibc-devel  # Fedora
sudo apt install build-essential  # Ubuntu
```

### Runtime Issues

**Library not found:**
```bash
# Check library exists
ls -la dist/libparacast.so

# Check executable RPATH
readelf -d dist/test_gpu_cpu | grep RPATH

# Run from dist/ directory
cd dist && ./test_gpu_cpu
```

**GPU initialization fails:**
```bash
# Check GPU support
lspci | grep -E "(VGA|3D|Display)"

# Check drivers
nvidia-smi        # NVIDIA
rocm-smi         # AMD  
intel_gpu_top    # Intel
```

**Shader compilation errors:**
- Usually indicates type information loss in JSON serialization
- Use `Paracast_CreateLargeNetwork()` instead of `LoadFromJSON()` 
- Verify model was saved with correct float32 type

### Performance Issues

**Poor GPU speedup:**
- Network too small (use larger/deeper networks)
- Batch size too small (increase sample count)  
- GPU overhead dominates (try CPU-only for small workloads)

**High memory usage:**
- Reduce batch size in `ForwardBatch` calls
- Free networks with `Paracast_Free()` when done
- Monitor GPU memory with `nvidia-smi`

## Integration Examples

### Godot 4 (GDScript)
```gdscript
# Use GDExtension to call C functions
var paracast = load("res://libparacast.gdextension")
var network = paracast.create_large_network()
paracast.init_gpu(network)
```

### Unity (C#)
```csharp
[DllImport("libparacast")]
public static extern long Paracast_CreateLargeNetwork();

[DllImport("libparacast")] 
public static extern int Paracast_InitGPU(long handle);
```

### Python (ctypes)
```python
import ctypes
lib = ctypes.CDLL('./libparacast.so')

lib.Paracast_CreateLargeNetwork.restype = ctypes.c_longlong
network = lib.Paracast_CreateLargeNetwork()
lib.Paracast_InitGPU(network)
```

## License

Apache 2.0 - Same as Paragon AI framework

## Contributing

1. Test changes with both benchmark programs
2. Verify GPU and CPU paths work correctly  
3. Update API documentation for new functions
4. Ensure cross-platform compatibility

For issues or questions, check error output from `Paracast_GetLastError()`.




GPU v CPU:
./model_benchmark --model mnist_model.json --cpu-only
./model_benchmark --model mnist_model.json --gpu-only
Model Benchmark Tool
Model: mnist_model.json
Samples per run: 500
Number of runs: 3

Loading model...
Model loaded successfully
Generating test data...

=== CPU Benchmark (3 runs of 500 samples) ===
  Run 1/3: 0.820s (609.8 samples/sec)
  Run 2/3: 0.888s (563.0 samples/sec)
  Run 3/3: 1.028s (486.6 samples/sec)
  Average: 0.912s (548.3 samples/sec)
  Range: 0.820s - 1.028s
  Total time: 2.736s

=== Summary ===
CPU: 3/3 successful runs, avg 548.3 samples/sec
Model Benchmark Tool
Model: mnist_model.json
Samples per run: 500
Number of runs: 3

Loading model...
Model loaded successfully
Generating test data...

=== GPU Benchmark (3 runs of 500 samples) ===
[wgpu] [Warn] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.
🚀 GPU Selected: 0x25a2 (0x10de) - Type: discrete-gpu
GPU warmup...
  Run 1/3: 0.112s (4474.1 samples/sec)
  Run 2/3: 0.110s (4563.0 samples/sec)
  Run 3/3: 0.098s (5108.9 samples/sec)
  Average: 0.106s (4699.2 samples/sec)
  Range: 0.098s - 0.112s
  Total time: 0.319s

=== Summary ===
GPU: 3/3 successful runs, avg 4699.2 samples/sec
