#include "entropy_calculator_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MIN_WINDOW_SIZE 16
#define MAX_WINDOW_SIZE 1024
#define DEFAULT_BUFFER_SIZE (1024 * 1024) // 1 MB buffer
#define WINDOW_HALF_SIZE 10  // Будем использовать 10 точек слева и 10 справа

// Structure for temporary computation data
typedef struct {
    size_t window_size;
    size_t step;
    uint32_t *histogram;
    size_t current_offset;
    size_t points_written;
    size_t points_to_write;
    
    // Buffers for filtering
    double *entropy_norm_buffer;
    size_t *offset_buffer;
    size_t buffer_capacity;
    size_t buffer_size;
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
    
    // Initialize buffers
    ctx->buffer_capacity = max_points > 0 ? max_points : 1000;
    ctx->entropy_norm_buffer = (double*)malloc(ctx->buffer_capacity * sizeof(double));
    ctx->offset_buffer = (size_t*)malloc(ctx->buffer_capacity * sizeof(size_t));
    ctx->buffer_size = 0;

    if (!ctx->histogram || !ctx->entropy_norm_buffer || !ctx->offset_buffer) {
        free(ctx->histogram);
        free(ctx->entropy_norm_buffer);
        free(ctx->offset_buffer);
        free(ctx);
        return NULL;
    }

    return ctx;
}

// Free context
static void free_window_context(WindowContext *ctx) {
    if (ctx) {
        free(ctx->histogram);
        free(ctx->entropy_norm_buffer);
        free(ctx->offset_buffer);
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

    if (entropy < 0.0) entropy = 0.0;
    return entropy;
}

// Compute maximum possible entropy for given window
static double max_entropy_for_window(size_t window_size) {
    if (window_size <= 256) {
        return log2((double)window_size);
    } else {
        return 8.0; // log2(256)
    }
}

// Новый фильтр с окном 10 точек и логарифмированием
static double apply_log_window_filter(const double *values, size_t index, size_t total) {
    const int HALF_WINDOW = WINDOW_HALF_SIZE; // 10 точек слева и 10 справа
    
    // Нужно минимум 2*HALF_WINDOW + 1 точек для полного окна
    if (total < 2 * HALF_WINDOW + 1) {
        // Если точек недостаточно, используем меньший размер окна
        int available_half = (total - 1) / 2;
        if (available_half == 0) {
            // Если совсем мало точек, возвращаем 0
            return 0.0;
        }
        
        // Используем доступный размер окна
        int start_idx = index - available_half;
        int end_idx = index + available_half;
        
        if (start_idx < 0) {
            end_idx += -start_idx;
            start_idx = 0;
        }
        if (end_idx >= (int)total) {
            start_idx -= (end_idx - total + 1);
            end_idx = total - 1;
        }
        if (start_idx < 0) start_idx = 0;
        
        // Вычисляем среднее в левой и правой половинах
        double left_sum = 0.0, right_sum = 0.0;
        int left_count = 0, right_count = 0;
        
        for (int i = start_idx; i < index; i++) {
            if (i >= 0 && i < (int)total) {
                left_sum += values[i];
                left_count++;
            }
        }
        
        for (int i = index + 1; i <= end_idx; i++) {
            if (i >= 0 && i < (int)total) {
                right_sum += values[i];
                right_count++;
            }
        }
        
        if (left_count == 0 || right_count == 0) {
            return 0.0;
        }
        
        double left_avg = left_sum / left_count;
        double right_avg = right_sum / right_count;
        double diff = right_avg - left_avg;
        
        // Применяем логарифмирование с сохранением знака
        double abs_diff = fabs(diff);
        if (abs_diff < 1e-10) {
            return 0.0;
        }
        
        // Логарифмируем разность (усиливаем большие скачки)
        // Используем log1p для устойчивости: log(1 + x)
        double log_value = log1p(abs_diff * 100.0); // Масштабируем для усиления
        
        // Возвращаем со знаком исходной разности
        return (diff >= 0) ? log_value : -log_value;
    }
    
    // Проверяем, что у нас достаточно точек для полного окна
    if (index < HALF_WINDOW || index >= total - HALF_WINDOW) {
        // Если точка слишком близко к краю, используем асимметричное окно
        int left_half = index;
        int right_half = total - index - 1;
        
        if (left_half > HALF_WINDOW) left_half = HALF_WINDOW;
        if (right_half > HALF_WINDOW) right_half = HALF_WINDOW;
        
        if (left_half == 0 || right_half == 0) {
            return 0.0;
        }
        
        double left_sum = 0.0, right_sum = 0.0;
        
        for (int i = 1; i <= left_half; i++) {
            left_sum += values[index - i];
        }
        
        for (int i = 1; i <= right_half; i++) {
            right_sum += values[index + i];
        }
        
        double left_avg = left_sum / left_half;
        double right_avg = right_sum / right_half;
        double diff = right_avg - left_avg;
        
        // Применяем логарифмирование
        double abs_diff = fabs(diff);
        if (abs_diff < 1e-10) {
            return 0.0;
        }
        
        double log_value = log1p(abs_diff * 100.0);
        return (diff >= 0) ? log_value : -log_value;
    }
    
    // Полное окно: 10 точек слева, 10 точек справа
    double left_sum = 0.0;
    for (int i = 1; i <= HALF_WINDOW; i++) {
        left_sum += values[index - i];
    }
    
    double right_sum = 0.0;
    for (int i = 1; i <= HALF_WINDOW; i++) {
        right_sum += values[index + i];
    }
    
    double left_avg = left_sum / HALF_WINDOW;
    double right_avg = right_sum / HALF_WINDOW;
    double diff = right_avg - left_avg;
    
    // Логарифмируем с сохранением знака
    double abs_diff = fabs(diff);
    if (abs_diff < 1e-10) {
        return 0.0;
    }
    
    // Усиливаем большие скачки логарифмом
    // log(1 + 100*x) дает:
    // x=0.1 -> log(11) ≈ 2.40
    // x=0.3 -> log(31) ≈ 3.43  
    // x=0.5 -> log(51) ≈ 3.93
    // x=0.9 -> log(91) ≈ 4.51
    // Разница между 0.1 и 0.9 примерно в 1.88 раз, но абсолютные значения больше
    
    double log_value = log1p(abs_diff * 100.0);
    return (diff >= 0) ? log_value : -log_value;
}

// Альтернативный вариант: фильтр с экспоненциальным усилением
static double apply_exp_window_filter(const double *values, size_t index, size_t total) {
    const int HALF_WINDOW = 10;
    
    if (index < HALF_WINDOW || index >= total - HALF_WINDOW) {
        return 0.0;
    }
    
    double left_sum = 0.0, right_sum = 0.0;
    
    for (int i = 1; i <= HALF_WINDOW; i++) {
        left_sum += values[index - i];
        right_sum += values[index + i];
    }
    
    double left_avg = left_sum / HALF_WINDOW;
    double right_avg = right_sum / HALF_WINDOW;
    double diff = right_avg - left_avg;
    
    // Экспоненциальное усиление: e^(k*x) - 1
    // При k=10: e^(10*0.1)=e^1≈2.718, e^(10*0.9)=e^9≈8103
    // Разница огромная!
    const double K = 5.0; // Коэффициент усиления
    
    if (diff >= 0) {
        return exp(K * diff) - 1.0;
    } else {
        return -(exp(-K * diff) - 1.0);
    }
}

// Комбинированный фильтр: логарифмирование + квадратичное усиление
static double apply_combined_filter(const double *values, size_t index, size_t total) {
    const int HALF_WINDOW = 10;
    
    if (total < 21) { // Нужно минимум 21 точка для полного окна
        return 0.0;
    }
    
    if (index < HALF_WINDOW || index >= total - HALF_WINDOW) {
        return 0.0;
    }
    
    // Вычисляем средние с весами (близкие точки важнее)
    double left_weighted = 0.0, right_weighted = 0.0;
    double left_weight_sum = 0.0, right_weight_sum = 0.0;
    
    for (int i = 1; i <= HALF_WINDOW; i++) {
        // Вес уменьшается с расстоянием (экспоненциально)
        double weight = exp(-i / 3.0);
        
        left_weighted += values[index - i] * weight;
        left_weight_sum += weight;
        
        right_weighted += values[index + i] * weight;
        right_weight_sum += weight;
    }
    
    double left_avg = left_weighted / left_weight_sum;
    double right_avg = right_weighted / right_weight_sum;
    double diff = right_avg - left_avg;
    
    // Комбинированное преобразование:
    // 1. Берем абсолютное значение
    double abs_diff = fabs(diff);
    
    if (abs_diff < 1e-10) {
        return 0.0;
    }
    
    // 2. Возводим в степень для усиления больших скачков
    double powered = pow(abs_diff, 1.5); // Степень 1.5 дает хорошее усиление
    
    // 3. Логарифмируем для сжатия динамического диапазона
    double result = log1p(powered * 10.0);
    
    // 4. Возвращаем со знаком
    return (diff >= 0) ? result : -result;
}

// Основной фильтр (выбираем комбинированный)
static double apply_main_filter(const double *values, size_t index, size_t total) {
    return apply_combined_filter(values, index, total);
}

static int process_block_for_context(WindowContext *ctx,
                                    const uint8_t *data,
                                    size_t data_size,
                                    size_t block_start,
                                    FILE *output) {
    double max_entropy = max_entropy_for_window(ctx->window_size);

    if (ctx->current_offset == 0) {
        memset(ctx->histogram, 0, 256 * sizeof(uint32_t));
        for (size_t i = 0; i < ctx->window_size && i < data_size; i++) {
            ctx->histogram[data[i]]++;
        }

        double entropy = entropy_from_histogram(ctx->histogram, ctx->window_size);
        double normalized_entropy = (max_entropy > 0) ? entropy / max_entropy : 0.0;

        if (!isfinite(normalized_entropy) || normalized_entropy < 0) {
            normalized_entropy = 0.0;
        } else if (normalized_entropy > 1.0) {
            normalized_entropy = 1.0;
        }

        if (ctx->buffer_size < ctx->buffer_capacity) {
            ctx->entropy_norm_buffer[ctx->buffer_size] = normalized_entropy;
            ctx->offset_buffer[ctx->buffer_size] = block_start;
            ctx->buffer_size++;
        }
        
        ctx->points_written++;
        ctx->current_offset = ctx->step;
    }

    size_t data_pos = ctx->current_offset;
    while (data_pos + ctx->window_size <= data_size && ctx->points_written < ctx->points_to_write) {

        if (data_pos >= ctx->step) {
            size_t remove_start = data_pos - ctx->step;
            size_t add_start = data_pos + ctx->window_size - ctx->step;

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
            memset(ctx->histogram, 0, 256 * sizeof(uint32_t));
            for (size_t i = 0; i < ctx->window_size; i++) {
                if (data_pos + i < data_size) {
                    ctx->histogram[data[data_pos + i]]++;
                }
            }
        }

        double entropy = entropy_from_histogram(ctx->histogram, ctx->window_size);
        double normalized_entropy = (max_entropy > 0) ? entropy / max_entropy : 0.0;

        if (!isfinite(normalized_entropy) || normalized_entropy < 0) {
            normalized_entropy = 0.0;
        } else if (normalized_entropy > 1.0) {
            normalized_entropy = 1.0;
        }

        if (ctx->buffer_size < ctx->buffer_capacity) {
            ctx->entropy_norm_buffer[ctx->buffer_size] = normalized_entropy;
            ctx->offset_buffer[ctx->buffer_size] = block_start + data_pos;
            ctx->buffer_size++;
        }

        ctx->points_written++;
        data_pos += ctx->step;
    }

    if (data_size >= ctx->window_size) {
        ctx->current_offset = data_pos - (data_size - ctx->window_size);
    } else {
        ctx->current_offset = 0;
    }

    if (ctx->current_offset > ctx->step) {
        ctx->current_offset = 0;
    }

    return 0;
}

static void write_filtered_data(WindowContext *ctx, FILE *output) {
    if (ctx->buffer_size == 0) {
        return;
    }
    
    for (size_t i = 0; i < ctx->buffer_size; i++) {
        double normalized_entropy = ctx->entropy_norm_buffer[i];
        
        // Применяем комбинированный фильтр
        double filtered_value = apply_main_filter(ctx->entropy_norm_buffer, i, ctx->buffer_size);
        
        // Абсолютное значение
        double filtered_abs = fabs(filtered_value);
        
        fprintf(output, "%zu,%zu,%zu,%.6f,%.6f,%.6f\n", 
                ctx->window_size, 
                ctx->step, 
                ctx->offset_buffer[i],
                normalized_entropy,
                filtered_value,
                filtered_abs);
    }
}

// Main streaming analysis function
int analyze_file_stream(const char *filename, const AnalysisConfig *config, FILE *output_file) {
    if (!filename || !config || !output_file) {
        return -1;
    }

    FILE *input_file = fopen(filename, "rb");
    if (!input_file) {
        perror("Error opening input file");
        return -1;
    }

    fseek(input_file, 0, SEEK_END);
    size_t file_size = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);

    printf("Analyzing file: %s\n", filename);
    printf("File size: %zu bytes\n", file_size);

    size_t buffer_size = config->buffer_size > 0 ? config->buffer_size : DEFAULT_BUFFER_SIZE;
    buffer_size = (buffer_size / 4096) * 4096;

    WindowContext **contexts = NULL;
    size_t num_contexts = 0;

    fprintf(output_file, "window_size,step,offset,entropy_norm,filtered_value,filtered_abs\n");

    // Create contexts
    for (size_t w_idx = 0; w_idx < config->num_windows; w_idx++) {
        size_t window_size = config->window_sizes[w_idx];
        if (window_size > file_size) continue;

        size_t steps[4];
        size_t num_steps;

        if (config->detailed_mode) {
            steps[0] = 1;
            steps[1] = window_size / 4;
            steps[2] = window_size / 2;
            steps[3] = window_size;
            num_steps = 4;

            if (steps[1] == 0) steps[1] = 1;
            if (steps[2] == 0) steps[2] = 1;
        } else {
            steps[0] = window_size / 4;
            if (steps[0] == 0) steps[0] = 1;
            num_steps = 1;
        }

        for (size_t s_idx = 0; s_idx < num_steps; s_idx++) {
            size_t step = steps[s_idx];

            size_t max_points = (file_size - window_size) / step + 1;
            if (config->max_points_per_window > 0 && max_points > config->max_points_per_window) {
                max_points = config->max_points_per_window;
            }

            WindowContext *ctx = create_window_context(window_size, step, max_points);
            if (!ctx) {
                for (size_t i = 0; i < num_contexts; i++) {
                    free_window_context(contexts[i]);
                }
                free(contexts);
                fclose(input_file);
                return -1;
            }

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
    printf("Using combined filter (10-point window + logarithmic enhancement)\n");

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

        for (size_t i = 0; i < num_contexts; i++) {
            if (contexts[i]->points_written < contexts[i]->points_to_write) {
                process_block_for_context(contexts[i], buffer, bytes_read, total_bytes_read, output_file);
            }
        }

        total_bytes_read += bytes_read;
        block_number++;

        if (block_number % 10 == 0) {
            printf("Processed: %.1f MB / %.1f MB\r", total_bytes_read / (1024.0 * 1024.0),
                   file_size / (1024.0 * 1024.0));
            fflush(stdout);
        }
    }

    printf("\nAnalysis complete. Processed %zu blocks\n", block_number);

    printf("Applying combined filter (10-point window + log enhancement)...\n");
    for (size_t i = 0; i < num_contexts; i++) {
        write_filtered_data(contexts[i], output_file);
    }

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

    size_t capacity = 8;
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