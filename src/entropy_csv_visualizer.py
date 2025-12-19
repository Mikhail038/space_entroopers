#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
import os
import sys
from pathlib import Path
import warnings
from scipy.ndimage import label
import json
from scipy.signal import find_peaks
warnings.filterwarnings('ignore')

@dataclass
class EntropySeries:
    """Структура для хранения ряда энтропии с улучшенной фильтрацией"""
    window_size: int
    step: int
    offsets: np.ndarray
    entropy_norm: np.ndarray
    filtered_value: np.ndarray  # Отфильтрованное значение (со знаком)
    filtered_abs: np.ndarray    # Абсолютное значение (сила скачка)
    filename: str = ""

    def __post_init__(self):
        if not isinstance(self.offsets, np.ndarray):
            self.offsets = np.array(self.offsets)
        if not isinstance(self.entropy_norm, np.ndarray):
            self.entropy_norm = np.array(self.entropy_norm)
        if not isinstance(self.filtered_value, np.ndarray):
            self.filtered_value = np.array(self.filtered_value)
        if not isinstance(self.filtered_abs, np.ndarray):
            self.filtered_abs = np.array(self.filtered_abs)

        self.entropy_norm = np.clip(self.entropy_norm, 0.0, 1.0)
        
        # Вычисляем производные
        if len(self.entropy_norm) > 1:
            self.gradient = np.gradient(self.entropy_norm)
            # Вторая производная для лучшего выделения границ
            self.second_derivative = np.gradient(self.gradient)
            
            # Находим пики в отфильтрованном сигнале
            self._find_peaks()
        else:
            self.gradient = np.array([])
            self.second_derivative = np.array([])
            self.peaks = np.array([], dtype=bool)
            self.peak_values = np.array([])
    
    def _find_peaks(self, prominence: float = 0.1):
        """Находит пики в отфильтрованном сигнале"""
        if len(self.filtered_abs) < 3:
            self.peaks = np.zeros(len(self.filtered_abs), dtype=bool)
            self.peak_values = np.array([])
            self.peak_positions = np.array([])
            return
        
        try:
            # Ищем пики в абсолютном значении
            peaks, properties = find_peaks(self.filtered_abs, 
                                          prominence=prominence,
                                          distance=max(1, self.window_size // self.step))
            
            self.peaks = np.zeros(len(self.filtered_abs), dtype=bool)
            self.peaks[peaks] = True
            self.peak_values = self.filtered_abs[peaks]
            self.peak_positions = self.offsets[peaks]
        except:
            # Если find_peaks не сработал, используем простой метод
            self.peaks = np.zeros(len(self.filtered_abs), dtype=bool)
            self.peak_values = np.array([])
            self.peak_positions = np.array([])

class CSVEntropyAnalyzer:
    @staticmethod
    def load_from_csv(csv_file: Union[str, Path]) -> Dict[int, List[EntropySeries]]:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV файл не найден: {csv_file}")

        print(f"Загрузка данных из CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            raise ValueError(f"Ошибка чтения CSV файла: {e}")

        print(f"Загружено {len(df)} записей")

        required_columns = ['window_size', 'step', 'offset', 'entropy_norm', 'filtered_value', 'filtered_abs']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}. "
                               f"Найдены колонки: {list(df.columns)}")

        grouped_results = {}

        for (window, step), group in df.groupby(['window_size', 'step']):
            if window not in grouped_results:
                grouped_results[window] = []

            series = EntropySeries(
                window_size=int(window),
                step=int(step),
                offsets=group['offset'].values,
                entropy_norm=group['entropy_norm'].values,
                filtered_value=group['filtered_value'].values,
                filtered_abs=group['filtered_abs'].values,
                filename=str(csv_path)
            )
            grouped_results[window].append(series)

        for window in grouped_results:
            # Сортируем серии по шагу (от меньшего к большему для лучшей детализации)
            grouped_results[window].sort(key=lambda x: x.step)

        print(f"Создано {sum(len(v) for v in grouped_results.values())} рядов энтропии")
        print(f"Размеры окон: {list(grouped_results.keys())}")
        
        for window in grouped_results:
            steps = [s.step for s in grouped_results[window]]
            print(f"  Окно {window}: шаги {steps}")
        
        return grouped_results

    @staticmethod
    def compare_multiple_csv(csv_files: List[Union[str, Path]]) -> Dict[str, Dict[int, List[EntropySeries]]]:
        results = {}
        for csv_file in csv_files:
            name = Path(csv_file).stem
            results[name] = CSVEntropyAnalyzer.load_from_csv(csv_file)

        return results

class EnhancedEntropyVisualizer:
    def __init__(self, results: Union[Dict[int, List[EntropySeries]],
                                      Dict[str, Dict[int, List[EntropySeries]]]]):
        self.results = results
        self.is_comparison = isinstance(next(iter(results.values())), dict)
        self.colors = plt.cm.tab10.colors
    
    def _get_best_series(self, results: Dict[int, List[EntropySeries]]) -> Optional[EntropySeries]:
        """
        Находит наилучшую серию для анализа:
        1. Сначала пробуем наименьшее окно (максимальная детализация)
        2. Потом выбираем серию с наименьшим шагом
        """
        if not results:
            return None
        
        # Пробуем найти серию с наименьшим окном
        smallest_window = min(results.keys())
        
        if smallest_window not in results or not results[smallest_window]:
            # Если нет данных для наименьшего окна, пробуем следующее
            available_windows = sorted(results.keys())
            for window in available_windows:
                if window in results and results[window]:
                    smallest_window = window
                    break
        
        if smallest_window not in results or not results[smallest_window]:
            return None
        
        # Выбираем серию с наименьшим шагом для лучшей детализации
        series_list = results[smallest_window]
        best_series = min(series_list, key=lambda x: x.step)
        
        print(f"Выбрана серия: окно={best_series.window_size}, шаг={best_series.step}, "
              f"точек={len(best_series.offsets)}")
        
        return best_series
    
    def plot_simple_detection(self,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (20, 8),
                              threshold_factor: float = 0.5,
                              show_details: bool = True) -> Optional[plt.Figure]:
        """
        Упрощенная визуализация: только энтропия и логарифмированная сила скачков
        
        Args:
            save_path: Путь для сохранения
            figsize: Размер фигуры
            threshold_factor: Коэффициент для определения порога (относительно максимума)
            show_details: Показывать детальную информацию о скачках
        """
        if self.is_comparison:
            print("Внимание: функция plot_simple_detection работает только с одним файлом")
            return None
        
        results = self.results
        if not results:
            print("Нет данных для визуализации")
            return None
        
        # Находим наилучшую серию для анализа
        target_series = self._get_best_series(results)
        if target_series is None:
            print("Не удалось найти подходящую серию данных")
            return None
        
        # Автоматически определяем порог
        if len(target_series.filtered_abs) > 0:
            max_filtered = np.max(target_series.filtered_abs)
            threshold = max_filtered * threshold_factor
        else:
            threshold = 0.1
        
        print(f"\nАнализ серии:")
        print(f"  Размер окна: {target_series.window_size}")
        print(f"  Шаг: {target_series.step}")
        print(f"  Количество точек: {len(target_series.offsets)}")
        print(f"  Максимальное значение фильтра: {max_filtered:.3f}")
        print(f"  Порог обнаружения: {threshold:.3f}")
        
        # Находим пики выше порога
        strong_peaks = target_series.filtered_abs > threshold
        peak_indices = np.where(strong_peaks)[0]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. График энтропии
        ax1.plot(target_series.offsets, target_series.entropy_norm,
                color='blue', alpha=0.8, linewidth=1.5,
                label=f'Нормированная энтропия (окно={target_series.window_size}, шаг={target_series.step})')
        
        # Выделяем области с сильными скачками
        for idx in peak_indices:
            if idx < len(target_series.offsets):
                x_pos = target_series.offsets[idx]
                y_pos = target_series.entropy_norm[idx]
                
                # Определяем цвет в зависимости от направления
                direction = target_series.filtered_value[idx]
                color = 'green' if direction > 0 else 'red'
                
                ax1.axvline(x=x_pos, color=color, alpha=0.4, linestyle='--', linewidth=0.8)
                
                # Подписываем только очень сильные скачки
                if target_series.filtered_abs[idx] > threshold * 2:
                    ax1.annotate(f'↑{target_series.filtered_abs[idx]:.2f}' if direction > 0 
                                else f'↓{target_series.filtered_abs[idx]:.2f}',
                                xy=(x_pos, y_pos),
                                xytext=(5, 10),
                                textcoords='offset points',
                                fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))
        
        ax1.set_title('Нормированная энтропия с обнаруженными скачками', fontsize=12)
        ax1.set_ylabel('Энтропия', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_ylim(-0.05, 1.05)
        
        # 2. Логарифмированный график силы скачков
        if np.any(target_series.filtered_abs > 0):
            # Избегаем log(0)
            log_filtered = np.log1p(target_series.filtered_abs * 10)
            ax2.plot(target_series.offsets, log_filtered,
                    color='brown', alpha=0.8, linewidth=1.5,
                    label='log(1 + 10×сила)')
            
            # Порог в логарифмированной шкале
            log_threshold = np.log1p(threshold * 10)
            ax2.axhline(y=log_threshold, color='darkred', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Порог ({log_threshold:.2f})')
            
            # Закрашиваем области выше порога
            above_threshold = target_series.filtered_abs > threshold
            if np.any(above_threshold):
                # Находим непрерывные области
                regions = self._find_contiguous_regions(above_threshold)
                
                for start_idx, end_idx in regions:
                    if start_idx < len(target_series.offsets) and end_idx < len(target_series.offsets):
                        start_x = target_series.offsets[start_idx]
                        end_x = target_series.offsets[end_idx]
                        
                        ax2.fill_betweenx(y=[0, np.max(log_filtered) * 1.1],
                                         x1=start_x, x2=end_x,
                                         color='red', alpha=0.2)
        else:
            ax2.plot(target_series.offsets, target_series.filtered_abs,
                    color='brown', alpha=0.8, linewidth=1.5,
                    label='Сила скачка')
            ax2.axhline(y=threshold, color='darkred', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'Порог ({threshold:.2f})')
        
        ax2.set_title('Логарифмированная сила скачков (усилены большие изменения)', fontsize=12)
        ax2.set_ylabel('log(сила)', fontsize=10)
        ax2.set_xlabel('Смещение в памяти (байты)', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Выводим статистику по найденным скачкам
        if show_details and len(peak_indices) > 0:
            self._print_peak_statistics(target_series, peak_indices, threshold)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _find_contiguous_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Находит непрерывные регионы в булевом массиве"""
        regions = []
        start = None
        
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                regions.append((start, i-1))
                start = None
        
        if start is not None:
            regions.append((start, len(mask)-1))
        
        return regions
    
    def _print_peak_statistics(self, series, peak_indices, threshold):
        """Выводит статистику по найденным пикам"""
        print(f"\n{'='*80}")
        print("СТАТИСТИКА ОБНАРУЖЕННЫХ СКАЧКОВ ЭНТРОПИИ")
        print(f"{'='*80}")
        print(f"Всего найдено скачков выше порога {threshold:.3f}: {len(peak_indices)}")
        print(f"{'='*80}")
        
        if len(peak_indices) == 0:
            return
        
        # Сортируем пики по силе
        peak_strengths = series.filtered_abs[peak_indices]
        sorted_indices = np.argsort(peak_strengths)[::-1]
        
        print(f"{'№':<3} {'Смещение':<12} {'Сила':<10} {'Направление':<12} "
              f"{'Энтропия':<10} {'Δ энтропии':<12}")
        print(f"{'-'*80}")
        
        for i, idx in enumerate(sorted_indices[:20]):  # Показываем топ-20
            peak_idx = peak_indices[idx]
            
            # Определяем направление
            direction = series.filtered_value[peak_idx]
            direction_str = "↑ ВВЕРХ" if direction > 0 else "↓ ВНИЗ"
            
            # Энтропия в точке скачка
            current_entropy = series.entropy_norm[peak_idx]
            
            # Дельта энтропии (разница с предыдущей точкой)
            if peak_idx > 0:
                delta_entropy = series.entropy_norm[peak_idx] - series.entropy_norm[peak_idx-1]
            else:
                delta_entropy = 0
            
            print(f"{i+1:<3} 0x{series.offsets[peak_idx]:08X} "
                  f"{series.filtered_abs[peak_idx]:<10.3f} "
                  f"{direction_str:<12} "
                  f"{current_entropy:<10.3f} "
                  f"{delta_entropy:<12.3f}")
        
        print(f"{'='*80}")
        
        # Анализируем направление скачков
        directions = series.filtered_value[peak_indices]
        up_count = np.sum(directions > 0)
        down_count = np.sum(directions < 0)
        
        print(f"Направления скачков:")
        print(f"  Вверх (↑): {up_count} ({up_count/len(peak_indices)*100:.1f}%)")
        print(f"  Вниз (↓): {down_count} ({down_count/len(peak_indices)*100:.1f}%)")
        
        # Находим самые сильные скачки каждого типа
        if up_count > 0:
            up_indices = peak_indices[directions > 0]
            up_strengths = series.filtered_abs[up_indices]
            strongest_up_idx = up_indices[np.argmax(up_strengths)]
            print(f"\nСамый сильный скачок ВВЕРХ:")
            print(f"  Смещение: 0x{series.offsets[strongest_up_idx]:08X}")
            print(f"  Сила: {series.filtered_abs[strongest_up_idx]:.3f}")
            print(f"  Энтропия: {series.entropy_norm[strongest_up_idx]:.3f}")
        
        if down_count > 0:
            down_indices = peak_indices[directions < 0]
            down_strengths = series.filtered_abs[down_indices]
            strongest_down_idx = down_indices[np.argmax(down_strengths)]
            print(f"\nСамый сильный скачок ВНИЗ:")
            print(f"  Смещение: 0x{series.offsets[strongest_down_idx]:08X}")
            print(f"  Сила: {series.filtered_abs[strongest_down_idx]:.3f}")
            print(f"  Энтропия: {series.entropy_norm[strongest_down_idx]:.3f}")
        
        print(f"{'='*80}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Упрощенный визуализатор энтропии с логарифмической фильтрацией (2 графика)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # Базовый анализ
  %(prog)s results.csv
  
  # Анализ с настройкой порога
  %(prog)s results.csv --threshold 0.3
  
  # Анализ без детальной статистики
  %(prog)s results.csv --no-stats
  
  # Анализ с указанием конкретного окна
  %(prog)s results.csv --window-size 1024
        """
    )
    
    parser.add_argument('csv_file', help='CSV файл с результатами энтропийного анализа')
    parser.add_argument('-o', '--output', default='./entropy_report',
                       help='Директория для сохранения отчета')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Коэффициент порога (относительно максимума, 0-1)')
    parser.add_argument('--window-size', type=int, default=None,
                       help='Конкретный размер окна для анализа (если не указан, берется наименьшее)')
    parser.add_argument('--step', type=int, default=None,
                       help='Конкретный шаг для анализа (если не указан, берется наименьший)')
    parser.add_argument('--no-stats', action='store_true',
                       help='Не показывать детальную статистику')
    parser.add_argument('--no-gui', action='store_true',
                       help='Не показывать графики (только сохранять)')
    parser.add_argument('--simple', action='store_true', default=True,
                       help='Использовать упрощенный режим с 2 графиками (по умолчанию)')
    
    args = parser.parse_args()
    
    try:
        print(f"Анализ файла: {args.csv_file}")
        results = CSVEntropyAnalyzer.load_from_csv(args.csv_file)
        
        # Если указан конкретный размер окна
        if args.window_size is not None:
            if args.window_size not in results:
                print(f"Внимание: размер окна {args.window_size} не найден в данных.")
                print(f"Доступные размеры окон: {list(results.keys())}")
                # Используем наименьшее доступное окно
                args.window_size = min(results.keys())
                print(f"Будет использовано окно: {args.window_size}")
        
        visualizer = EnhancedEntropyVisualizer(results)
        
        # Создаем директорию для отчета
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Генерируем упрощенную визуализацию (2 графика)
        plot_path = output_dir / "simple_entropy_analysis.png"
        print("\nГенерация упрощенной визуализации (2 графика)...")
        
        fig = visualizer.plot_simple_detection(
            save_path=str(plot_path),
            threshold_factor=args.threshold,
            show_details=not args.no_stats
        )
        
        if fig is None:
            print("Не удалось создать график")
            return 1
        
        print(f"Графики сохранены: {plot_path}")
        
        # Сохраняем информацию о выбранных параметрах
        info_path = output_dir / "analysis_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Анализ файла: {args.csv_file}\n")
            f.write(f"Дата анализа: {pd.Timestamp.now()}\n")
            f.write(f"Коэффициент порога: {args.threshold}\n")
            f.write(f"Указанный размер окна: {args.window_size}\n")
            f.write(f"Указанный шаг: {args.step}\n")
            f.write(f"Режим: упрощенный (2 графика)\n\n")
            
            f.write("Доступные данные:\n")
            for window_size, series_list in results.items():
                steps = [s.step for s in series_list]
                f.write(f"  Окно {window_size}: шаги {steps}\n")
        
        print(f"Информация об анализе сохранена: {info_path}")
        
        # Показываем графики если нет флага --no-gui
        if not args.no_gui:
            try:
                plt.show()
            except:
                print("\nГрафики сохранены в файлы (нет доступного DISPLAY)")
        
        return 0
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())