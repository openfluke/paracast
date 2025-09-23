#!/bin/bash
set -e

echo "Building C ABI shared library..."
cd "$(dirname "$0")"

# Create dist directory
mkdir -p dist

# Build the shared library
CGO_ENABLED=1 go build -buildmode=c-shared -o dist/libparacast.so ./cabi

# Compile the test program
echo "Compiling test program..."
cd dist
gcc -O3 -o test_gpu_cpu test_gpu_cpu.c -I. -L. -lparacast -Wl,-rpath,'$ORIGIN'

echo "Build complete!"
echo "Run with: cd dist && ./test_gpu_cpu"