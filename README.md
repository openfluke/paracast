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


# Grid RL

sudo dnf install ncurses-devel

LD_LIBRARY_PATH=. ./grid_rl

gcc grid_rl.c -lncurses -L. -Wl,-rpath,'$ORIGIN' -lparacast -lm -o grid_rl


./grid_rl
gcc grid_rl_paragon_mt.c -I. -L. -lparacast -lncursesw -lm -pthread \
  -Wl,-rpath,'$ORIGIN' -o grid_rl_paragon_mt


./grid_rl_paragon_mt --workers 4 --episodes 1500 --start 4,0 --demoeps 8 --save grid_q_best.json



# Pendulum

gcc pendulum_tabular_paragon.c -I. -L. -lparacast -lm -o pendulum_tabular_paragon   -Wl,-rpath,'$ORIGIN'

./pendulum_tabular_paragon                     
./pendulum_tabular_paragon --episodes 3000     
./pendulum_tabular_paragon --gpu              
./pendulum_tabular_paragon --save qtable.csv --savefit fit.csv


Training tabular Q: episodes=1500 bins=19 use_gpu_fit=no
[train] ep  100/1500  eps=0.937  G=87.516
[train] ep  200/1500  eps=0.874  G=111.307
[train] ep  300/1500  eps=0.811  G=118.032
[train] ep  400/1500  eps=0.747  G=94.349
[train] ep  500/1500  eps=0.684  G=83.508
[train] ep  600/1500  eps=0.620  G=88.904
[train] ep  700/1500  eps=0.557  G=86.979
[train] ep  800/1500  eps=0.494  G=142.835
[train] ep  900/1500  eps=0.430  G=97.056
[train] ep 1000/1500  eps=0.367  G=86.853
[train] ep 1100/1500  eps=0.304  G=196.042
[train] ep 1200/1500  eps=0.240  G=199.989
[train] ep 1300/1500  eps=0.177  G=85.820
[train] ep 1400/1500  eps=0.113  G=199.494
[train] ep 1500/1500  eps=0.050  G=85.992
Greedy rollout: steps=200  return=142.420
Saved rollout to rollout.csv
Fitting Paragon NN to Q-table (supervised)…
[wgpu] [Warn] Detected skylake derivative running on mesa i915. Clears to srgb textures will use manual shader clears.
🚀 GPU Selected: 0x25a2 (0x10de) - Type: discrete-gpu
[fit] epoch   50  MSE≈0.000000 (Q units)
[fit] epoch  100  MSE≈0.000000 (Q units)
[fit] epoch  150  MSE≈0.000000 (Q units)
[fit] epoch  200  MSE≈0.000000 (Q units)
[fit] epoch  250  MSE≈0.000000 (Q units)
[fit] epoch  300  MSE≈0.000000 (Q units)
[fit] epoch  350  MSE≈0.000000 (Q units)
[fit] epoch  400  MSE≈0.000000 (Q units)
[fit] eval MSE≈0.000000 (original Q units) on 1024 random states
Avg return over last 50 episodes: 107.979





# Python example outputs

python paracast_demo.py

🔧 create model

▶️  forward (cpu) — before any training
forward time: 0.0036s
output (first 10): [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]  | argmax = 0

🎓 first train (cpu) — nudging to class 9, 10 epochs, LR=0.1
train time: 10.0457s
▶️  forward (cpu) — after first train
forward time: 0.0037s
output (first 10): [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.993]  | argmax = 9
Change in argmax: 0 -> 9

🎓 second train (cpu) — now nudging to class 0, 10 epochs, LR=0.1
train time: 10.0005s
▶️  forward (cpu) — after second train
forward time: 0.0036s
output (first 10): [0.993, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003]  | argmax = 0
Change in argmax: 9 -> 0

📊 Training summary:
  Total train time: 20.0462s
  Forward times: before=0.0036s, after1=0.0037s, after2=0.0036s

🚀 cpu stress (big batches)
cpu timing (multiple big batches)…
round 1: 278.1 samples/s  (batch time: 14.728s, measured ns: 14.728s)
round 2: 282.5 samples/s  (batch time: 14.497s, measured ns: 14.497s)
round 3: 287.8 samples/s  (batch time: 14.234s, measured ns: 14.234s)

Overall avg: 282.8 samples/s over 3 rounds

🖨️ sample outputs from big batch:
batch[0] (first 10): [0.993, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003]  | argmax = 0

Full output vectors comparison:
Before train: [1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
After first train: [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.993]
After second train: [0.993, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.003]
