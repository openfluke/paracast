#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "libparacast.h"

#define MAX_PATH 1024
#define DEFAULT_SAMPLES 500
#define DEFAULT_RUNS 3

typedef struct {
    double total_time;
    double min_time;
    double max_time;
    double avg_throughput;
    int successful_runs;
} benchmark_stats;

// Read file into string
char* read_file(const char* filepath) {
    FILE* file = fopen(filepath, "r");
    if (!file) {
        printf("Error: Cannot open file '%s'\n", filepath);
        return NULL;
    }
    
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* content = malloc(length + 1);
    if (!content) {
        fclose(file);
        return NULL;
    }
    
    fread(content, 1, length, file);
    content[length] = '\0';
    fclose(file);
    
    return content;
}

// Run benchmark for specified mode
benchmark_stats run_benchmark(long long network, float* test_data, int num_samples, int num_runs, const char* mode, int use_gpu) {
    benchmark_stats stats = {0};
    
    printf("\n=== %s Benchmark (%d runs of %d samples) ===\n", mode, num_runs, num_samples);
    
    float* outputs = malloc(num_samples * 10 * sizeof(float));
    if (!outputs) {
        printf("Memory allocation failed\n");
        return stats;
    }
    
    // Configure GPU/CPU mode
    if (use_gpu) {
        if (Paracast_InitGPU(network) == 0) {
            printf("GPU initialization failed, skipping GPU benchmark\n");
            free(outputs);
            return stats;
        }
        // Warmup GPU
        printf("GPU warmup...\n");
        Paracast_WarmupGPU(network, 20);
    } else {
        Paracast_DisableGPU(network);
    }
    
    stats.min_time = 1e9;  // Very large initial value
    
    for (int run = 0; run < num_runs; run++) {
        printf("  Run %d/%d: ", run + 1, num_runs);
        fflush(stdout);
        
        long long elapsed_ns = Paracast_ForwardBatch(network, test_data, num_samples, outputs);
        
        if (elapsed_ns > 0) {
            double elapsed_sec = elapsed_ns / 1e9;
            double throughput = num_samples / elapsed_sec;
            
            stats.total_time += elapsed_sec;
            if (elapsed_sec < stats.min_time) stats.min_time = elapsed_sec;
            if (elapsed_sec > stats.max_time) stats.max_time = elapsed_sec;
            stats.successful_runs++;
            
            printf("%.3fs (%.1f samples/sec)\n", elapsed_sec, throughput);
        } else {
            printf("FAILED\n");
        }
    }
    
    if (stats.successful_runs > 0) {
        double avg_time = stats.total_time / stats.successful_runs;
        stats.avg_throughput = num_samples / avg_time;
        
        printf("  Average: %.3fs (%.1f samples/sec)\n", avg_time, stats.avg_throughput);
        printf("  Range: %.3fs - %.3fs\n", stats.min_time, stats.max_time);
        printf("  Total time: %.3fs\n", stats.total_time);
    }
    
    free(outputs);
    return stats;
}

void print_usage(const char* prog) {
    printf("Usage: %s --model <path.json> [options]\n", prog);
    printf("Options:\n");
    printf("  --model <path>    Path to model JSON file (required)\n");
    printf("  --samples <n>     Number of samples per run (default: %d)\n", DEFAULT_SAMPLES);
    printf("  --runs <n>        Number of benchmark runs (default: %d)\n", DEFAULT_RUNS);
    printf("  --cpu-only        Only run CPU benchmark\n");
    printf("  --gpu-only        Only run GPU benchmark\n");
    printf("  --help            Show this help\n");
    printf("\nExample:\n");
    printf("  %s --model ./mnist_model.json --samples 1000 --runs 5\n", prog);
}

int main(int argc, char** argv) {
    const char* model_path = NULL;
    int num_samples = DEFAULT_SAMPLES;
    int num_runs = DEFAULT_RUNS;
    int cpu_only = 0;
    int gpu_only = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            num_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cpu-only") == 0) {
            cpu_only = 1;
        } else if (strcmp(argv[i], "--gpu-only") == 0) {
            gpu_only = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (!model_path) {
        printf("Error: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    
    if (num_samples <= 0 || num_runs <= 0) {
        printf("Error: samples and runs must be positive\n");
        return 1;
    }
    
    printf("Model Benchmark Tool\n");
    printf("Model: %s\n", model_path);
    printf("Samples per run: %d\n", num_samples);
    printf("Number of runs: %d\n", num_runs);
    
    // Load model
    printf("\nLoading model...\n");
    char* json_content = read_file(model_path);
    if (!json_content) {
        return 1;
    }
    
    long long network = Paracast_LoadFromJSON(json_content, 0); // Load without GPU initially
    free(json_content);
    
    if (network == 0) {
        printf("Failed to load model from JSON\n");
        return 1;
    }
    
    printf("Model loaded successfully\n");
    
    // Generate test data (assuming 28x28 MNIST-like input)
    printf("Generating test data...\n");
    float* test_data = malloc(num_samples * 784 * sizeof(float));
    if (!test_data) {
        printf("Memory allocation failed\n");
        Paracast_Free(network);
        return 1;
    }
    
    Paracast_GenerateTestData(test_data, num_samples, 42); // Fixed seed for reproducibility
    
    benchmark_stats cpu_stats = {0};
    benchmark_stats gpu_stats = {0};
    
    // Run CPU benchmark
    if (!gpu_only) {
        cpu_stats = run_benchmark(network, test_data, num_samples, num_runs, "CPU", 0);
    }
    
    // Run GPU benchmark  
    if (!cpu_only) {
        gpu_stats = run_benchmark(network, test_data, num_samples, num_runs, "GPU", 1);
    }
    
    // Comparison and summary
    if (!cpu_only && !gpu_only && cpu_stats.successful_runs > 0 && gpu_stats.successful_runs > 0) {
        printf("\n=== Performance Comparison ===\n");
        double speedup = cpu_stats.avg_throughput > 0 ? gpu_stats.avg_throughput / cpu_stats.avg_throughput : 0;
        
        printf("CPU avg throughput: %.1f samples/sec\n", cpu_stats.avg_throughput);
        printf("GPU avg throughput: %.1f samples/sec\n", gpu_stats.avg_throughput);
        printf("GPU speedup: %.2fx\n", speedup);
        
        double cpu_avg_time = cpu_stats.total_time / cpu_stats.successful_runs;
        double gpu_avg_time = gpu_stats.total_time / gpu_stats.successful_runs;
        double time_speedup = cpu_avg_time / gpu_avg_time;
        
        printf("Time speedup: %.2fx (CPU: %.3fs/run, GPU: %.3fs/run)\n", 
               time_speedup, cpu_avg_time, gpu_avg_time);
        
        if (speedup > 5.0) {
            printf("Result: GPU DOMINATES! Excellent acceleration\n");
        } else if (speedup > 2.0) {
            printf("Result: GPU provides significant acceleration\n");
        } else if (speedup > 1.2) {
            printf("Result: GPU provides modest improvement\n");
        } else if (speedup > 0.8) {
            printf("Result: GPU and CPU performance comparable\n");
        } else {
            printf("Result: CPU outperforms GPU (overhead too high)\n");
        }
        
        // Validate outputs match
        printf("\n=== Output Validation ===\n");
        float* cpu_output = malloc(num_samples * 10 * sizeof(float));
        float* gpu_output = malloc(num_samples * 10 * sizeof(float));
        
        if (cpu_output && gpu_output) {
            // Quick validation run
            Paracast_DisableGPU(network);
            Paracast_ForwardBatch(network, test_data, 10, cpu_output); // Just first 10 samples
            
            Paracast_InitGPU(network);
            Paracast_ForwardBatch(network, test_data, 10, gpu_output);
            
            float max_diff = Paracast_CompareOutputs(cpu_output, gpu_output, 100); // 10 samples * 10 outputs
            printf("Max difference (first 10 samples): %e\n", max_diff);
            
            if (max_diff < 1e-6) {
                printf("Accuracy: CPU and GPU outputs are numerically identical\n");
            } else if (max_diff < 1e-3) {
                printf("Accuracy: CPU and GPU outputs match within acceptable tolerance\n");
            } else {
                printf("Warning: Notable differences between CPU and GPU outputs\n");
            }
        }
        
        free(cpu_output);
        free(gpu_output);
    }
    
    printf("\n=== Summary ===\n");
    if (cpu_stats.successful_runs > 0) {
        printf("CPU: %d/%d successful runs, avg %.1f samples/sec\n", 
               cpu_stats.successful_runs, num_runs, cpu_stats.avg_throughput);
    }
    if (gpu_stats.successful_runs > 0) {
        printf("GPU: %d/%d successful runs, avg %.1f samples/sec\n", 
               gpu_stats.successful_runs, num_runs, gpu_stats.avg_throughput);
    }
    
    // Cleanup
    Paracast_Free(network);
    free(test_data);
    
    return 0;
}