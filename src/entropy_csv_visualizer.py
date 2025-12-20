import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EntropySeries:
    window_size: int
    step: int
    offsets: np.ndarray
    entropy_norm: np.ndarray
    filtered_value: np.ndarray
    filtered_abs: np.ndarray
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

        if len(self.entropy_norm) > 1:
            self.gradient = np.gradient(self.entropy_norm)
            self.second_derivative = np.gradient(self.gradient)
            self._find_peaks()
        else:
            self.gradient = np.array([])
            self.second_derivative = np.array([])
            self.peaks = np.array([], dtype=bool)
            self.peak_values = np.array([])

    def _find_peaks(self, prominence: float = 0.1):
        if len(self.filtered_abs) < 3:
            self.peaks = np.zeros(len(self.filtered_abs), dtype=bool)
            self.peak_values = np.array([])
            self.peak_positions = np.array([])
            return

        try:
            peaks, _ = find_peaks(self.filtered_abs,
                                 prominence=prominence,
                                 distance=max(1, self.window_size // self.step))
            self.peaks = np.zeros(len(self.filtered_abs), dtype=bool)
            self.peaks[peaks] = True
            self.peak_values = self.filtered_abs[peaks]
            self.peak_positions = self.offsets[peaks]
        except:
            self.peaks = np.zeros(len(self.filtered_abs), dtype=bool)
            self.peak_values = np.array([])
            self.peak_positions = np.array([])

class CSVEntropyAnalyzer:
    @staticmethod
    def load_from_csv(csv_file: Union[str, Path]) -> dict:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            raise FileNotFoundError(f"Файл не найден: {csv_file}")

        df = pd.read_csv(csv_file)
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
            grouped_results[window].sort(key=lambda x: x.step)

        return grouped_results

class EntropyVisualizer:
    def __init__(self, results: dict):
        self.results = results
        self.colors = plt.cm.tab10.colors

    def _get_top_series(self, n: int = 3) -> List[EntropySeries]:
        if not self.results:
            return []

        sorted_windows = sorted(self.results.keys())

        top_series = []
        for window in sorted_windows[:n]:
            if window in self.results and self.results[window]:
                series_list = self.results[window]
                best_series = min(series_list, key=lambda x: x.step)
                top_series.append(best_series)

        return top_series

    def _find_dense_peaks(self, offsets: np.ndarray, window_size: int, radius_factor: int = 10) -> np.ndarray:
        if len(offsets) == 0:
            return np.array([], dtype=bool)

        radius = window_size * radius_factor
        dense_mask = np.zeros(len(offsets), dtype=bool)

        for i, offset in enumerate(offsets):
            neighbors = np.sum(np.abs(offsets - offset) <= radius)
            if neighbors > 1:
                dense_mask[i] = True

        return dense_mask

    def plot_detection(self,
                      save_path: Optional[str] = None,
                      figsize: Tuple[int, int] = (20, 12),
                      threshold_factor: float = 0.5,
                      show_stats: bool = True) -> Optional[plt.Figure]:

        results = self.results
        if not results:
            return None

        top_series = self._get_top_series(3)
        if not top_series:
            return None

        target_series = top_series[0]

        if len(target_series.filtered_abs) > 0:
            max_filtered = np.max(target_series.filtered_abs)
            threshold = max_filtered * threshold_factor
        else:
            threshold = 0.1

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                                gridspec_kw={'height_ratios': [1, 1, 1, 1]})

        for i, series in enumerate(top_series):
            ax = axes[i]

            if len(series.filtered_abs) > 0:
                series_max_filtered = np.max(series.filtered_abs)
                series_threshold = series_max_filtered * threshold_factor
            else:
                series_threshold = 0.1

            strong_peaks = series.filtered_abs > series_threshold
            peak_indices = np.where(strong_peaks)[0]

            ax.plot(series.offsets, series.entropy_norm,
                   color='blue', alpha=0.8, linewidth=1.5)

            if len(peak_indices) > 0:
                peak_offsets = series.offsets[peak_indices]
                dense_mask = self._find_dense_peaks(peak_offsets, series.window_size, 10)

                for j, idx in enumerate(peak_indices):
                    if idx < len(series.offsets):
                        x_pos = series.offsets[idx]
                        if j < len(dense_mask) and dense_mask[j]:
                            color = 'purple'
                            alpha = 0.6
                        else:
                            direction = series.filtered_value[idx]
                            color = 'green' if direction > 0 else 'red'
                            alpha = 0.3
                        ax.axvline(x=x_pos, color=color, alpha=alpha,
                                  linestyle='--', linewidth=0.6)

            ax.set_ylabel(f'Энтропия\n(window={series.window_size})')
            ax.grid(True, alpha=0.2)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f'Window Size: {series.window_size}, Step: {series.step}')

        ax4 = axes[3]

        if np.any(target_series.filtered_abs > 0):
            log_filtered = np.log1p(target_series.filtered_abs * 10)
            ax4.plot(target_series.offsets, log_filtered,
                    color='brown', alpha=0.8, linewidth=1.5)

            log_threshold = np.log1p(threshold * 10)
            ax4.axhline(y=log_threshold, color='darkred', linestyle='--',
                       linewidth=1.2, alpha=0.6)

            above_threshold = target_series.filtered_abs > threshold
            if np.any(above_threshold):
                regions = self._find_contiguous_regions(above_threshold)
                for start_idx, end_idx in regions:
                    if start_idx < len(target_series.offsets) and end_idx < len(target_series.offsets):
                        start_x = target_series.offsets[start_idx]
                        end_x = target_series.offsets[end_idx]
                        ax4.fill_betweenx(y=[0, np.max(log_filtered) * 1.1],
                                         x1=start_x, x2=end_x,
                                         color='red', alpha=0.15)

        ax4.set_ylabel('log(сила)')
        ax4.set_xlabel('Смещение')
        ax4.grid(True, alpha=0.2)
        ax4.set_title(f'Сила скачков (Window Size: {target_series.window_size})')

        plt.tight_layout()

        if save_path:
            save_path = self._get_unique_filename(save_path)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def _get_unique_filename(self, filepath: str) -> str:
        path = Path(filepath)
        if not path.exists():
            return filepath

        stem = path.stem
        suffix = path.suffix
        directory = path.parent

        counter = 1
        while True:
            new_name = f"{stem}_{counter}{suffix}"
            new_path = directory / new_name
            if not new_path.exists():
                return str(new_path)
            counter += 1

    def _find_contiguous_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
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

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file')
    parser.add_argument('-o', '--output', default='./entropy_report')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--window-size', type=int, default=None)
    parser.add_argument('--no-stats', action='store_true')
    parser.add_argument('--no-gui', action='store_true')

    args = parser.parse_args()

    try:
        results = CSVEntropyAnalyzer.load_from_csv(args.csv_file)

        if args.window_size is not None:
            if args.window_size not in results:
                args.window_size = min(results.keys())

        visualizer = EntropyVisualizer(results)

        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)

        base_filename = "entropy_analysis.png"
        plot_path = output_dir / base_filename

        unique_plot_path = visualizer._get_unique_filename(str(plot_path))

        fig = visualizer.plot_detection(
            save_path=unique_plot_path,
            threshold_factor=args.threshold,
            show_stats=not args.no_stats
        )

        if fig is None:
            return 1

        if not args.no_gui:
            try:
                plt.show()
            except:
                pass

        return 0

    except Exception as e:
        return 1

if __name__ == "__main__":
    sys.exit(main())
