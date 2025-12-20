#include "entropy_calculator_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

size_t *parse_window_sizes(const char *str, size_t *num_sizes) {
    char *copy = strdup(str);
    char *token = strtok(copy, ",");

    size_t capacity = 10;
    size_t *sizes = (size_t *)malloc(capacity * sizeof(size_t));
    *num_sizes = 0;

    while (token) {
        if (*num_sizes >= capacity) {
            capacity *= 2;
            sizes = (size_t *)realloc(sizes, capacity * sizeof(size_t));
            if (!sizes) {
                free(copy);
                return NULL;
            }
        }

        char *endptr;
        long value = strtol(token, &endptr, 10);
        if (*endptr == '\0' && value > 0) {
            sizes[(*num_sizes)++] = (size_t)value;
        }

        token = strtok(NULL, ",");
    }

    free(copy);
    return sizes;
}

void print_help(const char *program_name) {
    printf("Usage: %s <input_file> [options]\n", program_name);
    printf("\nOptions:\n");
    printf("  -o <file.csv>    Output CSV file (default: stdout)\n");
    printf("  -w <sizes>       Window sizes as comma-separated list (e.g., 64,128,256)\n");
    printf("  -d               Detailed mode (all steps: 1, window/4, window/2, window)\n");
    printf("  -f               Fast mode (only step: window/4) [default]\n");
    printf("  -m <number>      Max points per window (0 = unlimited)\n");
    printf("  -b <size>        Buffer size in bytes (default: 1 MB)\n");
    printf("  -h               Show this help\n");
    printf("\nExamples:\n");
    printf("  %s firmware.bin -o results.csv -w 64,128,256 -d\n", program_name);
    printf("  %s largefile.bin -f -m 10000 -b 4194304\n", program_name);
    printf("  %s data.bin -d | head -1000  # Show first 1000 lines\n", program_name);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }
    }

    const char *input_filename = argv[1];
    const char *output_filename = NULL;

    AnalysisConfig config;
    config.window_sizes = NULL;
    config.num_windows = 0;
    config.detailed_mode = 0; // Fast mode by default
    config.max_points_per_window = 10000;
    config.buffer_size = 1024 * 1024; // 1 MB default

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_filename = argv[++i];
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            config.window_sizes = parse_window_sizes(argv[++i], &config.num_windows);
        } else if (strcmp(argv[i], "-d") == 0) {
            config.detailed_mode = 1;
        } else if (strcmp(argv[i], "-f") == 0) {
            config.detailed_mode = 0;
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            config.max_points_per_window = strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            config.buffer_size = strtoul(argv[++i], NULL, 10);
            if (config.buffer_size < 4096)
                config.buffer_size = 4096;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_help(argv[0]);
            return 1;
        }
    }

    // Open file to determine its size
    FILE *test_file = fopen(input_filename, "rb");
    if (!test_file) {
        perror("Error opening input file");
        return 1;
    }

    fseek(test_file, 0, SEEK_END);
    size_t file_size = ftell(test_file);
    fclose(test_file);

    printf("File: %s\n", input_filename);
    printf("Size: %.2f MB\n", file_size / (1024.0 * 1024.0));

    // If window sizes not specified, auto-select
    if (!config.window_sizes) {
        printf("Auto-selecting window sizes...\n");
        config.window_sizes = auto_window_sizes(file_size, &config.num_windows);

        if (!config.window_sizes) {
            printf("Error allocating memory for window sizes\n");
            return 1;
        }
    }

    printf("Window sizes: ");
    for (size_t i = 0; i < config.num_windows; i++) {
        printf("%zu ", config.window_sizes[i]);
    }
    printf("\n");

    printf("Mode: %s\n", config.detailed_mode ? "detailed" : "fast (step=window/4)");
    printf("Max points per window: %zu\n", config.max_points_per_window);
    printf("Buffer size: %.2f MB\n", config.buffer_size / (1024.0 * 1024.0));

    // Open output file or use stdout
    FILE *output_file = stdout;
    if (output_filename) {
        output_file = fopen(output_filename, "w");
        if (!output_file) {
            perror("Error opening output file");
            free_window_sizes(config.window_sizes);
            return 1;
        }
        printf("Results will be saved to: %s\n", output_filename);
    } else {
        printf("Results printed to screen (use -o to save to file)\n");
    }

    // Measure execution time
    clock_t start_time = clock();

    // Perform analysis
    int result = analyze_file_stream(input_filename, &config, output_file);

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    if (result == 0) {
        printf("\nAnalysis completed successfully in %.2f seconds\n", elapsed_time);
    } else {
        printf("\nError during analysis\n");
    }

    // Close files and free memory
    if (output_file != stdout) {
        fclose(output_file);
    }

    free_window_sizes(config.window_sizes);

    return result;
}
