#ifndef ENTROPY_CALCULATOR_STREAM_H
#define ENTROPY_CALCULATOR_STREAM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// Конфигурация анализа
typedef struct {
    size_t *window_sizes;
    size_t num_windows;
    int detailed_mode;
    size_t max_points_per_window;
    size_t buffer_size;  // Размер буфера для чтения/обработки
} AnalysisConfig;

// Основные функции
int analyze_file_stream(const char *filename, 
                       const AnalysisConfig *config,
                       FILE *output_file);

// Утилиты
size_t* auto_window_sizes(size_t file_size, size_t *num_sizes);
void free_window_sizes(size_t *sizes);

#endif