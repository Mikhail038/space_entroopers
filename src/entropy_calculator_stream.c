#include "entropy_calculator_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MIN_WINDOW_SIZE 16
#define MAX_WINDOW_SIZE 4096
#define DEFAULT_BUFFER_SIZE (1024 * 1024) // 1 MB buffer

// Structure for temporary computation data
typedef struct {
    size_t window_size;
    size_t step;
    uint32_t *histogram;
    size_t current_offset;
    size_t points_written;
    size_t points_to_write;
} WindowContext;

// Initialize window context
static WindowContext *create_window_context(size_t window_size, size_t step, size_t max_points) {
    WindowContext *ctx = (WindowContext*)calloc(1, sizeof(WindowContext));
    if (!ctx) return NULL;

    ctx->window_size = window_size;
    ctx->step = step;
    ctx->histogram = (uint32_t*)calloc(256, sizeof(uint32_t));
    ctx->current_offset = 0;
    ctx->points_written = 0;
    ctx->points_to_write = max_points;

    if (!ctx->histogram) {
        free(ctx);
        return NULL;
    }

    return ctx;
}

// Free context
static void free_window_context(WindowContext *ctx) {
    if (ctx) {
        free(ctx->histogram);
        free(ctx);
    }
}

// Fixed entropy_from_histogram function
static double entropy_from_histogram(const uint32_t *hist, size_t window_size) {
    double entropy = 0.0;
    double inv_window_size = 1.0 / window_size;

    for (int i = 0; i < 256; i++) {
        if (hist[i] > 0) {
            double p = hist[i] * inv_window_size;
            entropy -= p * log2(p);
        }
    }

    // Ensure entropy is non-negative
    if (entropy < 0.0) entropy = 0.0;

    return entropy;
}

// NEW function: compute maximum possible entropy for given window
static double max_entropy_for_window(size_t window_size) {
    // Maximum entropy is achieved with uniform distribution
    // If window_size <= 256, maximum = log2(window_size)
    // If window_size > 256, maximum = log2(256) = 8
    if (window_size <= 256) {
        return log2((double)window_size);
    } else {
        return 8.0; // log2(256)
    }
}

static int process_block_for_context(WindowContext *ctx,
                                    const uint8_t *data,
                                    size_t data_size,
                                    size_t block_start,
                                    FILE *output) {
    // Compute maximum entropy for this window size
    double max_entropy = max_entropy_for_window(ctx->window_size);

    // If this is the first block for this context
    if (ctx->current_offset == 0) {
        // Initialize histogram with first window
        memset(ctx->histogram, 0, 256 * sizeof(uint32_t));
        for (size_t i = 0; i < ctx->window_size && i < data_size; i++) {
            ctx->histogram[data[i]]++;
        }

        // Compute entropy and normalize it
        double entropy = entropy_from_histogram(ctx->histogram, ctx->window_size);
        double normalized_entropy = (max_entropy > 0) ? entropy / max_entropy : 0.0;

        fprintf(output, "%zu,%zu,%zu,%.6f,%.6f\n", ctx->window_size, ctx->step, block_start, entropy,
                normalized_entropy); // Add normalized entropy
        ctx->points_written++;
        ctx->current_offset = ctx->step;
    }

    // Process remaining points in the block
    size_t data_pos = ctx->current_offset;
    while (data_pos + ctx->window_size <= data_size && ctx->points_written < ctx->points_to_write) {

        // Check if we have data to remove
        if (data_pos >= ctx->step) {
            size_t remove_start = data_pos - ctx->step;
            size_t add_start = data_pos + ctx->window_size - ctx->step;

            // Safely update histogram
            for (size_t i = 0; i < ctx->step; i++) {
                if (remove_start + i < data_size) {
                    uint8_t old_byte = data[remove_start + i];
                    if (ctx->histogram[old_byte] > 0) {
                        ctx->histogram[old_byte]--;
                    }
                }
                if (add_start + i < data_size) {
                    uint8_t new_byte = data[add_start + i];
                    ctx->histogram[new_byte]++;
                }
            }
        } else {
            // If data_pos < step, recalculate histogram from scratch
            // (this can happen at block boundaries)
            memset(ctx->histogram, 0, 256 * sizeof(uint32_t));
            for (size_t i = 0; i < ctx->window_size; i++) {
                if (data_pos + i < data_size) {
                    ctx->histogram[data[data_pos + i]]++;
                }
            }
        }

        // Compute entropy
        double entropy = entropy_from_histogram(ctx->histogram, ctx->window_size);
        double normalized_entropy = (max_entropy > 0) ? entropy / max_entropy : 0.0;

        // Check for NaN or infinity
        if (!isfinite(normalized_entropy) || normalized_entropy < 0) {
            normalized_entropy = 0.0;
        } else if (normalized_entropy > 1.0) {
            normalized_entropy = 1.0; // Cap at 1.0
        }

        fprintf(output, "%zu,%zu,%zu,%.6f,%.6f\n", ctx->window_size, ctx->step, block_start + data_pos,
                entropy, normalized_entropy);

        ctx->points_written++;
        data_pos += ctx->step;
    }

    // Save current position for next block
    if (data_size >= ctx->window_size) {
        ctx->current_offset = data_pos - (data_size - ctx->window_size);
    } else {
        ctx->current_offset = 0;
    }

    // Ensure offset doesn't exceed step
    if (ctx->current_offset > ctx->step) {
        ctx->current_offset = 0;
    }

    return 0;
}

// Main streaming analysis function
int analyze_file_stream(const char *filename, const AnalysisConfig *config, FILE *output_file) {
    if (!filename || !config || !output_file) {
        return -1;
    }

    // Open file for reading
    FILE *input_file = fopen(filename, "rb");
    if (!input_file) {
        perror("Error opening input file");
        return -1;
    }

    // Get file size
    fseek(input_file, 0, SEEK_END);
    size_t file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    printf("Analyzing file: %s\n", filename);
    printf("File size: %zu bytes\n", file_size);

    // Determine buffer size
    size_t buffer_size = config->buffer_size > 0 ? config->buffer_size : DEFAULT_BUFFER_SIZE;
    buffer_size = (buffer_size / 4096) * 4096; // Align to 4K

    // Create contexts for all window/step combinations
    WindowContext **contexts = NULL;
    size_t num_contexts = 0;

    // Prepare CSV header
    fprintf(output_file, "window_size,step,offset,entropy,entropy_norm\n");

    // Create contexts
    for (size_t w_idx = 0; w_idx < config->num_windows; w_idx++) {
        size_t window_size = config->window_sizes[w_idx];
        if (window_size > file_size) continue;

        // Determine steps
        size_t steps[4];
        size_t num_steps;

        if (config->detailed_mode) {
            steps[0] = 1;
            steps[1] = window_size / 4;
            steps[2] = window_size / 2;
            steps[3] = window_size;
            num_steps = 4;

            // Correct zero steps
            if (steps[1] == 0) steps[1] = 1;
            if (steps[2] == 0) steps[2] = 1;
        } else {
            steps[0] = window_size / 4;
            if (steps[0] == 0) steps[0] = 1;
            num_steps = 1;
        }

        // Create context for each step
        for (size_t s_idx = 0; s_idx < num_steps; s_idx++) {
            size_t step = steps[s_idx];

            // Calculate number of points for this step
            size_t max_points = (file_size - window_size) / step + 1;
            if (config->max_points_per_window > 0 && max_points > config->max_points_per_window) {
                max_points = config->max_points_per_window;
            }

            // Create context
            WindowContext *ctx = create_window_context(window_size, step, max_points);
            if (!ctx) {
                // Free already created contexts
                for (size_t i = 0; i < num_contexts; i++) {
                    free_window_context(contexts[i]);
                }
                free(contexts);
                fclose(input_file);
                return -1;
            }

            // Add context to array
            contexts = (WindowContext **)realloc(contexts, (num_contexts + 1) * sizeof(WindowContext *));
            contexts[num_contexts++] = ctx;
        }
    }

    if (num_contexts == 0) {
        printf("No valid window/step combinations\n");
        fclose(input_file);
        free(contexts);
        return 0;
    }

    printf("Created %zu analysis contexts\n", num_contexts);

    // Read and process file in blocks
    uint8_t *buffer = (uint8_t*)malloc(buffer_size);
    if (!buffer) {
        perror("Error allocating buffer");
        for (size_t i = 0; i < num_contexts; i++) {
            free_window_context(contexts[i]);
        }
        free(contexts);
        fclose(input_file);
        return -1;
    }

    size_t total_bytes_read = 0;
    size_t block_number = 0;

    while (total_bytes_read < file_size) {
        size_t bytes_to_read = buffer_size;
        if (total_bytes_read + bytes_to_read > file_size) {
            bytes_to_read = file_size - total_bytes_read;
        }

        size_t bytes_read = fread(buffer, 1, bytes_to_read, input_file);
        if (bytes_read == 0) {
            if (ferror(input_file)) {
                perror("Error reading file");
                break;
            }
            break;
        }

        // Process block for each context
        for (size_t i = 0; i < num_contexts; i++) {
            if (contexts[i]->points_written < contexts[i]->points_to_write) {
                process_block_for_context(contexts[i], buffer, bytes_read, total_bytes_read, output_file);
            }
        }

        total_bytes_read += bytes_read;
        block_number++;

        // Display progress
        if (block_number % 10 == 0) {
            printf("Processed: %.1f MB / %.1f MB\r", total_bytes_read / (1024.0 * 1024.0),
                   file_size / (1024.0 * 1024.0));
            fflush(stdout);
        }
    }

    printf("\nAnalysis complete. Processed %zu blocks\n", block_number);

    // Free resources
    for (size_t i = 0; i < num_contexts; i++) {
        free_window_context(contexts[i]);
    }
    free(contexts);
    free(buffer);
    fclose(input_file);

    return 0;
}

// Automatic window size selection
size_t* auto_window_sizes(size_t file_size, size_t *num_sizes) {
    size_t max_window = file_size / 4;
    if (max_window > MAX_WINDOW_SIZE) max_window = MAX_WINDOW_SIZE;
    if (max_window < MIN_WINDOW_SIZE) max_window = MIN_WINDOW_SIZE;

    size_t capacity = 16;
    size_t *sizes = (size_t*)malloc(capacity * sizeof(size_t));
    *num_sizes = 0;

    size_t window = MIN_WINDOW_SIZE;
    while (window <= max_window) {
        if (*num_sizes >= capacity) {
            capacity *= 2;
            sizes = (size_t*)realloc(sizes, capacity * sizeof(size_t));
            if (!sizes) {
                return NULL;
            }
        }

        sizes[(*num_sizes)++] = window;
        window = (size_t)(window * 2);
    }

    return sizes;
}

void free_window_sizes(size_t *sizes) {
    free(sizes);
}
