#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "libparacast.h"

#define NUM_SAMPLES 1000

int main() {
    printf("Creating large network for GPU vs CPU comparison...\n");
    
    // Create network
    long long network = Paracast_CreateLargeNetwork();
    if (network == 0) {
        printf("Failed to create network\n");
        return 1;
    }
    
    // Allocate memory for test data
    float* testData = (float*)malloc(NUM_SAMPLES * 784 * sizeof(float));
    float* cpuOutputs = (float*)malloc(NUM_SAMPLES * 10 * sizeof(float));
    float* gpuOutputs = (float*)malloc(NUM_SAMPLES * 10 * sizeof(float));
    
    if (!testData || !cpuOutputs || !gpuOutputs) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Generate test data
    printf("Generating %d test samples...\n", NUM_SAMPLES);
    Paracast_GenerateTestData(testData, NUM_SAMPLES, 42);
    
    // CPU Test
    printf("\n=== CPU Test ===\n");
    Paracast_DisableGPU(network);
    
    long long cpuTime = Paracast_ForwardBatch(network, testData, NUM_SAMPLES, cpuOutputs);
    double cpuSeconds = cpuTime / 1e9;
    double cpuThroughput = NUM_SAMPLES / cpuSeconds;
    
    printf("CPU Time: %.3f seconds\n", cpuSeconds);
    printf("CPU Throughput: %.2f samples/sec\n", cpuThroughput);
    
    // GPU Test
    printf("\n=== GPU Test ===\n");
    if (Paracast_InitGPU(network) == 0) {
        printf("GPU initialization failed\n");
        return 1;
    }
    
    // GPU Warmup
    printf("GPU warmup...\n");
    long long warmupTime = Paracast_WarmupGPU(network, 50);
    printf("GPU Warmup Time: %.3f seconds\n", warmupTime / 1e9);
    
    long long gpuTime = Paracast_ForwardBatch(network, testData, NUM_SAMPLES, gpuOutputs);
    double gpuSeconds = gpuTime / 1e9;
    double gpuThroughput = NUM_SAMPLES / gpuSeconds;
    
    printf("GPU Time: %.3f seconds\n", gpuSeconds);
    printf("GPU Throughput: %.2f samples/sec\n", gpuThroughput);
    
    // Performance Comparison
    printf("\n=== Performance Comparison ===\n");
    double speedup = cpuSeconds / gpuSeconds;
    printf("GPU Speedup: %.2fx\n", speedup);
    
    if (speedup > 5.0) {
        printf("🚀 GPU DOMINATES! %.2fx faster than CPU!\n", speedup);
    } else if (speedup > 2.0) {
        printf("🔥 GPU is significantly faster: %.2fx speedup\n", speedup);
    } else if (speedup > 1.0) {
        printf("✅ GPU is faster: %.2fx speedup\n", speedup);
    } else {
        printf("❌ CPU is faster: %.2fx\n", 1.0/speedup);
    }
    
    double timeSaved = cpuSeconds - gpuSeconds;
    printf("Time saved by using GPU: %.3f seconds (%.1f%% reduction)\n", 
           timeSaved, (timeSaved/cpuSeconds)*100);
    
    // Output Validation
    printf("\n=== Output Validation ===\n");
    float maxDiff = Paracast_CompareOutputs(cpuOutputs, gpuOutputs, NUM_SAMPLES * 10);
    printf("Max difference between CPU/GPU: %e\n", maxDiff);
    
    // Show sample outputs
    printf("\n=== Sample Outputs (first 3 samples, all 10 values) ===\n");
    for (int i = 0; i < 3; i++) {
        printf("Sample %d:\n", i+1);
        printf("  CPU: ");
        for (int j = 0; j < 10; j++) {
            printf("%.4f ", cpuOutputs[i*10 + j]);
        }
        printf("\n  GPU: ");
        for (int j = 0; j < 10; j++) {
            printf("%.4f ", gpuOutputs[i*10 + j]);
        }
        printf("\n");
    }
    
    // Final verdict
    printf("\n=== FINAL VERDICT ===\n");
    if (maxDiff < 1e-5) {
        printf("✅ ACCURACY: CPU and GPU outputs are numerically identical\n");
    } else if (maxDiff < 1e-3) {
        printf("✅ ACCURACY: CPU and GPU outputs match within tolerance\n");
    } else {
        printf("⚠️  ACCURACY: Notable differences between CPU and GPU\n");
    }
    
    if (speedup > 5.0) {
        printf("🔥 PERFORMANCE: GPU acceleration is EXCELLENT!\n");
    } else if (speedup > 2.0) {
        printf("✅ PERFORMANCE: GPU acceleration working well\n");
    } else {
        printf("👍 PERFORMANCE: GPU provides modest improvement\n");
    }
    
    // Cleanup
    Paracast_Free(network);
    free(testData);
    free(cpuOutputs);
    free(gpuOutputs);
    
    return 0;
}