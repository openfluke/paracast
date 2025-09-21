# Paracast — C ABI Bindings for Paragon AI

Paracast is a lightweight Go project that exposes the Paragon AI framework as a C-shared library (.so/.dll). It enables calling Paragon’s neural network APIs from C, C++, or any engine that supports native libraries (e.g., Godot 4, Unity, Unreal).

## Goals
- Make Paragon’s GPU-agnostic AI (WebGPU/Vulkan) usable in external runtimes without rewriting models.
- Provide a portable bridge for simulation engines to train, run, and persist models directly.

## Features
- Create networks from layer definitions (width × height), activation functions, and connectivity flags.
- Load/Save networks via JSON (using Paragon’s persistence).
- Forward pass on float32 tensors.
- Read outputs into C arrays.
- Train on mini-batches, with optional GPU acceleration.
- Cross-compile to Linux (.so) and Windows (.dll) with auto-generated headers.

## Repo Layout
```
paracast/
├── cabi/                  # Go sources for the C ABI layer
│   ├── cabi.go            # Exported functions (//export ...)
│   └── main.go            # Empty main() (required for c-shared)
├── dist/                  # Build outputs
│   ├── libparacast.so     # Linux shared lib
│   ├── paracast.dll       # Windows DLL
│   ├── *.h                # Auto-generated C headers
│   ├── test.c / win_test.c # Example C clients
│   └── windows_demo/      # Runtime DLLs for Windows test
├── go.mod / go.sum        # Go module files
```

## Build Instructions

### Prerequisites
- Fedora (or any Linux with GCC, MinGW for cross-compilation).
- Go ≥ 1.22 with CGO enabled.
- gcc, glibc-devel, kernel-headers, and optionally mingw64-gcc.

### Linux (.so)
```bash
cd paracast
CGO_ENABLED=1 go build -buildmode=c-shared -o dist/libparacast.so ./cabi
```
Produces `libparacast.so` and `libparacast.h`.

### Windows (.dll)
```bash
cd paracast
export CC=x86_64-w64-mingw32-gcc
export CXX=x86_64-w64-mingw32-g++
CGO_ENABLED=1 go build -buildmode=c-shared -o dist/paracast.dll ./cabi
unset CC CXX
```
Produces `paracast.dll` and `paracast.h`.  
Requires MinGW runtime DLLs (`libgcc_s_seh-1.dll`, `libstdc++-6.dll`, `libwinpthread-1.dll`), included in `dist/windows_demo/`.

## Example (C)
```c
#include "libparacast.h"
#include <stdio.h>

int main() {
    long long net = Paracast_NewNetwork(...);

    float input[784];  // example: MNIST sample
    Paracast_Forward(net, input, 28, 28);

    float output[10];
    int n = Paracast_GetOutput(net, output, 10);

    printf("Got %d outputs, first=%f\n", n, output[0]);
    Paracast_Free(net);
}
```

### Build & Run
```bash
gcc test.c -I. -L. -lparacast -Wl,-rpath,'$ORIGIN' -o test
./test
```

## Integration with Godot 4
- Place `paracast.dll` (Windows) or `libparacast.so` (Linux) in your Godot project.
- Use GDNative / GDExtension to call exported functions (`Paracast_NewNetwork`, `Paracast_Forward`, etc.).
- Enables Godot scripts or C# code to drive Paragon networks in your game or simulation.

## Roadmap
- Ready-made Godot 4 GDExtension wrapper.
- Unity & Unreal plugin stubs.
- High-level bindings for Python, Rust, and C#.
- Additional training utilities (mini-batch, shuffle, Adam optimizer).

## License
Apache 2.0 — same as Paragon.  
This repo is designed to remain open and forkable, serving as the glue layer between engines and the Paragon AI framework.