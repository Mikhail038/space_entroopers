#include "entropy_calculator_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Парсинг размеров окон из строки
size_t* parse_window_sizes(const char *str, size_t *num_sizes) {
    char *copy = strdup(str);
    char *token = strtok(copy, ",");
    
    size_t capacity = 10;
    size_t *sizes = (size_t*)malloc(capacity * sizeof(size_t));
    *num_sizes = 0;
    
    while (token) {
        if (*num_sizes >= capacity) {
            capacity *= 2;
            sizes = (size_t*)realloc(sizes, capacity * sizeof(size_t));
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

// Вывод справки
void print_help(const char *program_name) {
    printf("Использование: %s <входной_файл> [опции]\n", program_name);
    printf("\nОпции:\n");
    printf("  -o <файл.csv>    Выходной CSV файл (по умолчанию: вывод на экран)\n");
    printf("  -w <размеры>     Размеры окон через запятую (например: 64,128,256)\n");
    printf("  -d               Детальный режим (все шаги: 1, окно/4, окно/2, окно)\n");
    printf("  -f               Быстрый режим (только шаг: окно/4) [по умолчанию]\n");
    printf("  -m <число>       Максимальное точек на окно (0 = без ограничений)\n");
    printf("  -b <размер>      Размер буфера в байтах (по умолчанию: 1 МБ)\n");
    printf("  -h               Показать эту справку\n");
    printf("\nПримеры:\n");
    printf("  %s firmware.bin -o results.csv -w 64,128,256 -d\n", program_name);
    printf("  %s largefile.bin -f -m 10000 -b 4194304\n", program_name);
    printf("  %s data.bin -d | head -1000  # Показать первые 1000 строк\n", program_name);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return 1;
    }
    
    // Проверка на запрос справки
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_help(argv[0]);
            return 0;
        }
    }
    
    const char *input_filename = argv[1];
    const char *output_filename = NULL;
    
    AnalysisConfig config = {0};
    config.detailed_mode = 0;  // По умолчанию быстрый режим
    config.max_points_per_window = 10000;
    config.buffer_size = 1024 * 1024;  // 1 МБ по умолчанию
    
    // Парсинг аргументов
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
            if (config.buffer_size < 4096) config.buffer_size = 4096;
        } else {
            printf("Неизвестный аргумент: %s\n", argv[i]);
            print_help(argv[0]);
            return 1;
        }
    }
    
    // Открываем файл для чтения, чтобы определить его размер
    FILE *test_file = fopen(input_filename, "rb");
    if (!test_file) {
        perror("Ошибка открытия входного файла");
        return 1;
    }
    
    fseek(test_file, 0, SEEK_END);
    size_t file_size = ftell(test_file);
    fclose(test_file);
    
    printf("Файл: %s\n", input_filename);
    printf("Размер: %.2f МБ\n", file_size / (1024.0 * 1024.0));
    
    // Если размеры окон не заданы, подбираем автоматически
    if (!config.window_sizes) {
        printf("Автоматический подбор размеров окон...\n");
        config.window_sizes = auto_window_sizes(file_size, &config.num_windows);
        
        if (!config.window_sizes) {
            printf("Ошибка выделения памяти для размеров окон\n");
            return 1;
        }
    }
    
    printf("Размеры окон: ");
    for (size_t i = 0; i < config.num_windows; i++) {
        printf("%zu ", config.window_sizes[i]);
    }
    printf("\n");
    
    printf("Режим: %s\n", config.detailed_mode ? "детальный" : "быстрый (шаг=окно/4)");
    printf("Макс точек на окно: %zu\n", config.max_points_per_window);
    printf("Размер буфера: %.2f МБ\n", config.buffer_size / (1024.0 * 1024.0));
    
    // Открываем выходной файл или используем stdout
    FILE *output_file = stdout;
    if (output_filename) {
        output_file = fopen(output_filename, "w");
        if (!output_file) {
            perror("Ошибка открытия выходного файла");
            free_window_sizes(config.window_sizes);
            return 1;
        }
        printf("Результаты будут сохранены в: %s\n", output_filename);
    } else {
        printf("Результаты выводятся на экран (используйте -o для сохранения в файл)\n");
    }
    
    // Замеряем время выполнения
    clock_t start_time = clock();
    
    // Выполняем анализ
    int result = analyze_file_stream(input_filename, &config, output_file);
    
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    if (result == 0) {
        printf("\nАнализ успешно завершен за %.2f секунд\n", elapsed_time);
    } else {
        printf("\nОшибка при выполнении анализа\n");
    }
    
    // Закрываем файлы и освобождаем память
    if (output_file != stdout) {
        fclose(output_file);
    }
    
    free_window_sizes(config.window_sizes);
    
    return result;
}