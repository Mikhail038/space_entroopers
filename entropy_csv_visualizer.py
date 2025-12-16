# #!/usr/bin/env python3

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
warnings.filterwarnings('ignore')

@dataclass
class EntropySeries:
    """Структура для хранения ряда энтропии (новый формат)"""
    window_size: int
    step: int
    offsets: np.ndarray
    entropy: np.ndarray
    entropy_norm: np.ndarray
    filename: str = ""
    
    def __post_init__(self):
        # Конвертируем в numpy массивы если нужно
        if not isinstance(self.offsets, np.ndarray):
            self.offsets = np.array(self.offsets)
        if not isinstance(self.entropy, np.ndarray):
            self.entropy = np.array(self.entropy)
        if not isinstance(self.entropy_norm, np.ndarray):
            self.entropy_norm = np.array(self.entropy_norm)
        
        # Ограничиваем значения [0, 1]
        self.entropy_norm = np.clip(self.entropy_norm, 0.0, 1.0)
        
        # Вычисляем производную (для поиска скачков)
        if len(self.entropy_norm) > 1:
            self.gradient = np.gradient(self.entropy_norm)
        else:
            self.gradient = np.array([])

class CSVEntropyAnalyzer:
    """Анализатор, работающий с CSV файлами нового формата"""
    
    @staticmethod
    def load_from_csv(csv_file: Union[str, Path]) -> Dict[int, List[EntropySeries]]:
        """
        Загружает результаты анализа из CSV файла (новый формат)
        
        Args:
            csv_file: Путь к CSV файлу с результатами
            
        Returns:
            Словарь: {размер_окна: [список EntropySeries]}
        """
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV файл не найден: {csv_file}")
        
        print(f"Загрузка данных из CSV: {csv_path}")
        
        # Читаем CSV файл
        df = pd.read_csv(csv_path)
        
        print(f"Загружено {len(df)} записей")
        
        # Проверяем наличие необходимых колонок (новый формат)
        required_columns = ['window_size', 'step', 'offset', 'entropy', 'entropy_norm']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}. "
                               f"Используйте новый формат: {', '.join(required_columns)}")
        
        # Группируем результаты
        grouped_results = {}
        
        for (window, step), group in df.groupby(['window_size', 'step']):
            if window not in grouped_results:
                grouped_results[window] = []
            
            series = EntropySeries(
                window_size=int(window),
                step=int(step),
                offsets=group['offset'].values,
                entropy=group['entropy'].values,
                entropy_norm=group['entropy_norm'].values,
                filename=str(csv_path)
            )
            grouped_results[window].append(series)
        
        # Сортируем серии внутри каждой группы по шагу (от большего к меньшему)
        for window in grouped_results:
            grouped_results[window].sort(key=lambda x: x.step, reverse=True)
        
        print(f"Создано {sum(len(v) for v in grouped_results.values())} рядов энтропии")
        return grouped_results
    
    @staticmethod
    def compare_multiple_csv(csv_files: List[Union[str, Path]]) -> Dict[str, Dict[int, List[EntropySeries]]]:
        """
        Загружает и сравнивает несколько CSV файлов
        
        Args:
            csv_files: Список путей к CSV файлам
            
        Returns:
            Словарь: {имя_файла: результаты}
        """
        results = {}
        for csv_file in csv_files:
            name = Path(csv_file).stem
            results[name] = CSVEntropyAnalyzer.load_from_csv(csv_file)
        
        return results

class EntropyVisualizer:
    """Универсальный визуализатор энтропии для нового формата CSV"""
    
    def __init__(self, results: Union[Dict[int, List[EntropySeries]], 
                                      Dict[str, Dict[int, List[EntropySeries]]]]):
        """
        Args:
            results: Может быть либо словарем результатов одного файла,
                    либо словарем сравнения нескольких файлов
        """
        self.results = results
        self.is_comparison = isinstance(next(iter(results.values())), dict)
        self.colors = plt.cm.tab10.colors
        
    def plot_single_file(self, 
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (16, 10),
                        show_jumps: bool = True,
                        jump_threshold: float = 0.3) -> plt.Figure:
        """
        Строит графики для одного файла
        
        Args:
            save_path: Путь для сохранения изображения
            figsize: Размер фигуры
            show_jumps: Показывать ли скачки энтропии
            jump_threshold: Порог для обнаружения скачков
            
        Returns:
            Объект фигуры matplotlib
        """
        if self.is_comparison:
            raise ValueError("Для сравнения файлов используйте plot_comparison")
        
        results = self.results
        num_windows = len(results)
        
        # Создаем сетку графиков
        fig, axes = plt.subplots(num_windows, 1, figsize=figsize, sharex=True)
        
        if num_windows == 1:
            axes = [axes]
        
        # Настраиваем отображение для каждого размера окна
        for ax, (window_size, series_list) in zip(axes, results.items()):
            # Рисуем каждую серию (шаг) разным цветом
            for i, series in enumerate(series_list):
                color = self.colors[i % len(self.colors)]
                
                # Определяем метку для шага
                if series.step == 1:
                    step_label = "шаг=1"
                elif series.step == window_size // 4:
                    step_label = "шаг=окно/4"
                elif series.step == window_size // 2:
                    step_label = "шаг=окно/2"
                elif series.step == window_size:
                    step_label = "шаг=окно"
                else:
                    step_label = f"шаг={series.step}"
                
                # Основная линия нормированной энтропии
                line = ax.plot(series.offsets, series.entropy_norm,
                       color=color, alpha=0.7, linewidth=1,
                       label=f'{step_label}')[0]
                
                # Показываем скачки энтропии
                if show_jumps and len(series.gradient) > 0:
                    jumps = np.where(np.abs(series.gradient) > jump_threshold)[0]
                    
                    if len(jumps) > 0:
                        # Группируем близкие скачки
                        clusters = self._cluster_indices(jumps, window_size)
                        
                        for cluster in clusters:
                            if len(cluster) > 0:
                                center_idx = cluster[len(cluster)//2]
                                x_pos = series.offsets[center_idx]
                                y_pos = series.entropy_norm[center_idx]
                                
                                # Аннотируем скачки
                                # if i == 0:  # Добавляем легенду только для первой серии
                                #     # ax.scatter(x_pos, y_pos, 
                                #     #          color='red', s=30, zorder=5,
                                #     #          marker='x', label='Скачок энтропии')
                                # else:
                                ax.scatter(x_pos, y_pos, 
                                            color='red', s=30, zorder=5,
                                            marker='x')
            
            # Настраиваем график
            ax.set_title(f'Размер окна: {window_size} байт', fontsize=11)
            ax.set_ylabel('Нормированная энтропия', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=9)
            
            # Устанавливаем пределы по Y с отступом от нуля
            # Собираем все значения нормированной энтропии для всех серий этого окна
            all_entropy_values = []
            for series in series_list:
                all_entropy_values.extend(series.entropy_norm)
            
            if all_entropy_values:
                min_val = np.min(all_entropy_values)
                max_val = np.max(all_entropy_values)
                
                # Создаем отступ снизу: 5% от диапазона, но не менее 0.05
                padding_bottom = max(0.05, (max_val - min_val) * 0.05)
                
                # Верхний предел с небольшим запасом
                padding_top = max(0.05, (max_val - min_val) * 0.05)
                
                # Нижняя граница: либо min_val - padding, либо -0.02 (чтобы ось X была видна)
                y_min = max(-0.02, min_val - padding_bottom)
                y_max = min(1.05, max_val + padding_top)
                
                ax.set_ylim(y_min, y_max)
            else:
                # Значения по умолчанию если нет данных
                ax.set_ylim(-0.05, 1.05)
            
            # Добавляем горизонтальную линию на уровне 0 для наглядности
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3, zorder=1)
            
            # Подписываем ось X только на последнем графике
            if ax == axes[-1]:
                ax.set_xlabel('Смещение в памяти (байты)', fontsize=10)
        
        # plt.suptitle('Анализ энтропии (нормированная)', fontsize=13, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_comparison(self,
                       save_path: Optional[str] = None,
                       figsize: Tuple[int, int] = (18, 12),
                       selected_window: Optional[int] = None) -> plt.Figure:
        """
        Сравнивает энтропию нескольких файлов
        
        Args:
            save_path: Путь для сохранения
            figsize: Размер фигуры
            selected_window: Конкретный размер окна для сравнения
            
        Returns:
            Объект фигуры matplotlib
        """
        if not self.is_comparison:
            raise ValueError("Для сравнения нужны данные нескольких файлов")
        
        # Выбираем общие размеры окон
        all_windows = set()
        for file_results in self.results.values():
            all_windows.update(file_results.keys())
        
        if selected_window is not None:
            windows_to_plot = [selected_window] if selected_window in all_windows else list(all_windows)[:1]
        else:
            windows_to_plot = sorted(all_windows)[:4]  # Ограничиваем 4 окнами
        
        num_windows = len(windows_to_plot)
        
        # Создаем сетку графиков
        fig, axes = plt.subplots(num_windows, 1, figsize=figsize, sharex=True)
        
        if num_windows == 1:
            axes = [axes]
        
        # Для каждого окна строим сравнение файлов
        for ax, window_size in zip(axes, windows_to_plot):
            # Собираем все данные для вычисления общего диапазона Y
            all_entropy_values = []
            
            for i, (file_name, file_results) in enumerate(self.results.items()):
                if window_size in file_results:
                    series_list = file_results[window_size]
                    
                    # Ищем серию с шагом window/4 (оптимальную)
                    target_series = None
                    for series in series_list:
                        if series.step == max(1, window_size // 4):
                            target_series = series
                            break
                    
                    if target_series is None:
                        continue
                    
                    color = self.colors[i % len(self.colors)]
                    
                    # Рисуем нормированную энтропию
                    line = ax.plot(target_series.offsets, target_series.entropy_norm,
                           color=color, alpha=0.7, linewidth=1,
                           label=file_name)[0]
                    
                    # Собираем значения энтропии для вычисления диапазона
                    all_entropy_values.extend(target_series.entropy_norm)
            
            ax.set_title(f'Сравнение файлов (окно: {window_size} байт)', fontsize=11)
            ax.set_ylabel('Нормированная энтропия', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=9)
            
            # Устанавливаем пределы по Y с отступом от нуля
            if all_entropy_values:
                min_val = np.min(all_entropy_values)
                max_val = np.max(all_entropy_values)
                
                # Создаем отступ снизу: 5% от диапазона, но не менее 0.05
                padding_bottom = max(0.05, (max_val - min_val) * 0.05)
                
                # Верхний предел с небольшим запасом
                padding_top = max(0.05, (max_val - min_val) * 0.05)
                
                # Нижняя граница с отступом
                y_min = max(-0.02, min_val - padding_bottom)
                y_max = min(1.05, max_val + padding_top)
                
                ax.set_ylim(y_min, y_max)
            else:
                ax.set_ylim(-0.05, 1.05)
            
            # Добавляем горизонтальную линию на уровне 0
            ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        
        axes[-1].set_xlabel('Смещение в памяти (байты)', fontsize=10)
        plt.suptitle('Сравнение нормированной энтропии', fontsize=13, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def detect_anomalies(self,
                        entropy_threshold: float = 0.9,
                        gradient_threshold: float = 0.3,
                        min_size: int = 32) -> List[Dict]:
        """
        Обнаружение аномалий в данных
        
        Args:
            entropy_threshold: Порог высокой энтропии (0-1)
            gradient_threshold: Порог для скачков энтропии
            min_size: Минимальный размер аномалии в байтах
            
        Returns:
            Список обнаруженных аномалий
        """
        anomalies = []
        
        # Для случая сравнения файлов анализируем каждый файл отдельно
        if self.is_comparison:
            for file_name, results in self.results.items():
                file_anomalies = self._analyze_single_file(results, entropy_threshold, 
                                                         gradient_threshold, min_size)
                for anomaly in file_anomalies:
                    anomaly['file'] = file_name
                    anomalies.append(anomaly)
        else:
            anomalies = self._analyze_single_file(self.results, entropy_threshold,
                                                gradient_threshold, min_size)
        
        return anomalies
    
    def _analyze_single_file(self, results, entropy_threshold, 
                           gradient_threshold, min_size) -> List[Dict]:
        """Анализ одного файла"""
        anomalies = []
        
        # Используем самое маленькое окно для максимальной детализации
        if not results:
            return anomalies
        
        smallest_window = min(results.keys())
        
        for window_size, series_list in results.items():
            if window_size != smallest_window:
                continue
                
            for series in series_list:
                # Используем серию с оптимальным шагом (window/4)
                if series.step == max(1, window_size // 4):
                    # 1. Обнаружение регионов с высокой энтропией
                    high_entropy_regions = self._find_high_entropy_regions(
                        series, entropy_threshold, min_size
                    )
                    
                    # 2. Обнаружение скачков энтропии
                    jumps = self._find_entropy_jumps(
                        series, gradient_threshold
                    )
                    
                    # 3. Обнаружение низкой энтропии
                    low_entropy_regions = self._find_low_entropy_regions(
                        series, 0.3, min_size
                    )
                    
                    # Объединяем все аномалии
                    for region in high_entropy_regions:
                        region['type'] = 'high_entropy'
                        region['window_size'] = window_size
                        region['step'] = series.step
                        anomalies.append(region)
                    
                    for jump in jumps:
                        jump['type'] = 'entropy_jump'
                        jump['window_size'] = window_size
                        jump['step'] = series.step
                        anomalies.append(jump)
                    
                    for region in low_entropy_regions:
                        region['type'] = 'low_entropy'
                        region['window_size'] = window_size
                        region['step'] = series.step
                        anomalies.append(region)
                    
                    break  # Используем только первую подходящую серию
        
        return anomalies
    
    def _find_high_entropy_regions(self, series, threshold, min_size):
        """Находит регионы с высокой энтропией"""
        regions = []
        high_entropy = series.entropy_norm > threshold
        
        if not np.any(high_entropy):
            return regions
        
        # Группируем последовательные точки
        labeled_array, num_features = label(high_entropy)
        
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            
            if len(indices) > 0:
                start_idx = indices[0]
                end_idx = indices[-1]
                
                start_offset = series.offsets[start_idx]
                end_offset = series.offsets[end_idx] + series.window_size
                region_size = end_offset - start_offset
                
                if region_size >= min_size:
                    region = {
                        'start': start_offset,
                        'end': end_offset,
                        'size': region_size,
                        'avg_entropy': np.mean(series.entropy_norm[indices]),
                        'max_entropy': np.max(series.entropy_norm[indices])
                    }
                    regions.append(region)
        
        return regions
    
    def _find_entropy_jumps(self, series, threshold):
        """Находит скачки энтропии"""
        jumps = []
        
        if len(series.gradient) == 0:
            return jumps
        
        # Находим точки где градиент превышает порог
        jump_indices = np.where(np.abs(series.gradient) > threshold)[0]
        
        # Группируем близкие скачки
        clusters = self._cluster_indices(jump_indices, series.window_size)
        
        for cluster in clusters:
            if len(cluster) > 0:
                center_idx = cluster[len(cluster)//2]
                jump = {
                    'position': series.offsets[center_idx],
                    'gradient': series.gradient[center_idx],
                    'entropy_before': series.entropy_norm[center_idx-1] if center_idx > 0 else 0,
                    'entropy_after': series.entropy_norm[center_idx] if center_idx < len(series.entropy_norm) else 0
                }
                jumps.append(jump)
        
        return jumps
    
    def _find_low_entropy_regions(self, series, threshold, min_size):
        """Находит регионы с низкой энтропией"""
        regions = []
        low_entropy = series.entropy_norm < threshold
        
        if not np.any(low_entropy):
            return regions
        
        # Группируем последовательные точки
        labeled_array, num_features = label(low_entropy)
        
        for i in range(1, num_features + 1):
            indices = np.where(labeled_array == i)[0]
            
            if len(indices) > 0:
                start_idx = indices[0]
                end_idx = indices[-1]
                
                start_offset = series.offsets[start_idx]
                end_offset = series.offsets[end_idx] + series.window_size
                region_size = end_offset - start_offset
                
                if region_size >= min_size:
                    region = {
                        'start': start_offset,
                        'end': end_offset,
                        'size': region_size,
                        'avg_entropy': np.mean(series.entropy_norm[indices]),
                        'min_entropy': np.min(series.entropy_norm[indices])
                    }
                    regions.append(region)
        
        return regions
    
    def _cluster_indices(self, indices, window_size):
        """Группирует близко расположенные индексы"""
        if len(indices) == 0:
            return []
        
        clusters = []
        current_cluster = [indices[0]]
        
        for idx in indices[1:]:
            if idx - current_cluster[-1] <= window_size:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        
        clusters.append(current_cluster)
        return clusters
    
    def generate_report(self,
                       output_dir: Union[str, Path] = "./entropy_report",
                       export_csv: bool = True) -> Dict:
        """
        Генерирует полный отчет об анализе
        
        Args:
            output_dir: Директория для сохранения отчета
            export_csv: Экспортировать ли аномалии в CSV
            
        Returns:
            Словарь с результатами анализа
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Обнаруживаем аномалии
        anomalies = self.detect_anomalies()
        
        # Создаем базовый отчет
        if self.is_comparison:
            report = {
                'analysis_type': 'comparison',
                'files': list(self.results.keys()),
                'total_anomalies': len(anomalies),
                'anomalies': anomalies,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        else:
            # Получаем информацию из первого ряда
            first_series = next(iter(next(iter(self.results.values()))))
            report = {
                'analysis_type': 'single_file',
                'source_file': first_series.filename,
                'window_sizes': list(self.results.keys()),
                'total_anomalies': len(anomalies),
                'anomalies': anomalies,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        
        # Экспортируем аномалии в CSV
        if export_csv and anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            
            # Добавляем информацию о файле для сравнения
            if self.is_comparison and 'file' in anomalies_df.columns:
                anomalies_df = anomalies_df[['file', 'type', 'start', 'end', 'size', 
                                           'avg_entropy', 'max_entropy', 'gradient']]
            
            csv_path = output_dir / "anomalies.csv"
            anomalies_df.to_csv(csv_path, index=False, encoding='utf-8')
            report['anomalies_csv'] = str(csv_path)
        
        # Сохраняем отчет в JSON
        json_path = output_dir / "report.json"
        
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=convert_numpy)
        
        # Генерируем графики (ТОЛЬКО ОСНОВНЫЕ, без тепловой карты)
        plot_path = output_dir / "entropy_analysis.png"
        
        if self.is_comparison:
            self.plot_comparison(save_path=str(plot_path))
        else:
            self.plot_single_file(save_path=str(plot_path))
        
        report['main_plot'] = str(plot_path)
        report['json_report'] = str(json_path)
        
        # Выводим сводку
        self._print_summary(report, output_dir)
        
        return report
    
    def _print_summary(self, report, output_dir):
        """Выводит сводку отчета"""
        print(f"\n{'='*60}")
        print("ОТЧЕТ АНАЛИЗА ЭНТРОПИИ (НОРМИРОВАННОЙ)")
        print(f"{'='*60}")
        
        if report['analysis_type'] == 'comparison':
            print(f"Сравнение {len(report['files'])} файлов:")
            for file in report['files']:
                print(f"  - {file}")
        else:
            print(f"Файл: {report.get('source_file', 'Неизвестно')}")
            print(f"Размеры окон: {report.get('window_sizes', [])}")
        
        print(f"\nОбнаружено аномалий: {report['total_anomalies']}")
        
        if report['total_anomalies'] > 0:
            # Группируем аномалии по типу
            anomalies = report['anomalies']
            type_counts = {}
            for anomaly in anomalies:
                a_type = anomaly.get('type', 'unknown')
                type_counts[a_type] = type_counts.get(a_type, 0) + 1
            
            print("По типам:")
            for a_type, count in type_counts.items():
                print(f"  - {a_type}: {count}")
            
            # Показываем самые крупные аномалии высокой энтропии
            high_entropy_anomalies = [a for a in anomalies if a.get('type') == 'high_entropy']
            if high_entropy_anomalies:
                sorted_by_size = sorted(high_entropy_anomalies, 
                                      key=lambda x: x.get('size', 0), 
                                      reverse=True)[:3]
                print("\nКрупнейшие регионы высокой энтропии (возможное шифрование):")
                for i, anomaly in enumerate(sorted_by_size, 1):
                    print(f"  {i}. Смещение: 0x{anomaly['start']:X}-0x{anomaly['end']:X} "
                          f"(размер: {anomaly['size']} байт, "
                          f"энтропия: {anomaly['avg_entropy']:.2f})")
        
        print(f"\nОтчеты сохранены в: {output_dir}")
        print(f"{'='*60}")

def main():
    """Главная функция командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Визуализатор энтропии из CSV файлов (новый формат)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Формат CSV: window_size,step,offset,entropy,entropy_norm

Примеры:
  # Анализ одного CSV файла
  %(prog)s results.csv
  
  # Анализ с сохранением отчета
  %(prog)s results.csv -o ./report/
  
  # Сравнение нескольких CSV файлов
  %(prog)s file1.csv file2.csv file3.csv --compare
  
  # Анализ с настройкой порогов
  %(prog)s results.csv --threshold 0.95 --min-size 64
  
  # Только обнаружение аномалий без графиков
  %(prog)s results.csv --detect-only --export-csv
        """
    )
    
    parser.add_argument('csv_files', nargs='+', 
                       help='CSV файлы с результатами энтропийного анализа')
    parser.add_argument('-o', '--output', default='./entropy_report',
                       help='Директория для сохранения отчета')
    parser.add_argument('--compare', action='store_true',
                       help='Сравнить несколько CSV файлов')
    parser.add_argument('--threshold', type=float, default=0.9,
                       help='Порог высокой энтропии (0-1)')
    parser.add_argument('--jump-threshold', type=float, default=0.3,
                       help='Порог для скачков энтропии')
    parser.add_argument('--min-size', type=int, default=32,
                       help='Минимальный размер аномалии (байты)')
    parser.add_argument('--detect-only', action='store_true',
                       help='Только обнаружение аномалий без графиков')
    parser.add_argument('--export-csv', action='store_true',
                       help='Экспортировать аномалии в CSV')
    parser.add_argument('--window', type=int,
                       help='Конкретный размер окна для анализа')
    parser.add_argument('--no-gui', action='store_true',
                       help='Не показывать графики (только сохранять)')
    
    args = parser.parse_args()
    
    try:
        # Загружаем данные из CSV (новый формат)
        if len(args.csv_files) == 1 and not args.compare:
            # Один файл
            print(f"Анализ файла: {args.csv_files[0]}")
            results = CSVEntropyAnalyzer.load_from_csv(args.csv_files[0])
            visualizer = EntropyVisualizer(results)
        else:
            # Несколько файлов для сравнения
            print(f"Сравнение {len(args.csv_files)} файлов...")
            results = CSVEntropyAnalyzer.compare_multiple_csv(args.csv_files)
            visualizer = EntropyVisualizer(results)
        
        # Обнаружение аномалий
        anomalies = visualizer.detect_anomalies(
            entropy_threshold=args.threshold,
            gradient_threshold=args.jump_threshold,
            min_size=args.min_size
        )
        
        print(f"Обнаружено аномалий: {len(anomalies)}")
        
        if args.detect_only:
            # Только обнаружение, без графиков
            if anomalies:
                print("\nОбнаруженные аномалии:")
                for i, anomaly in enumerate(anomalies, 1):
                    print(f"\n{i}. Тип: {anomaly.get('type', 'unknown')}")
                    if 'start' in anomaly:
                        print(f"   Смещение: 0x{anomaly['start']:X}-0x{anomaly['end']:X}")
                        print(f"   Размер: {anomaly['size']} байт")
                    if 'position' in anomaly:
                        print(f"   Позиция: 0x{anomaly['position']:X}")
                    if 'avg_entropy' in anomaly:
                        print(f"   Средняя энтропия: {anomaly['avg_entropy']:.3f}")
            
            # Экспорт в CSV если нужно
            if args.export_csv and anomalies:
                output_dir = Path(args.output)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                anomalies_df = pd.DataFrame(anomalies)
                csv_path = output_dir / "detected_anomalies.csv"
                anomalies_df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"\nАномалии сохранены в: {csv_path}")
            
            return 0
        
        # Полный отчет с графиками
        report = visualizer.generate_report(
            output_dir=args.output,
            export_csv=args.export_csv
        )
        
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