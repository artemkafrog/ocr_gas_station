import time
import psutil
import os
import Levenshtein
import pandas as pd

def is_predicted(predicted: str, expected: str) -> float:
    """Predictions for Accuracy"""
    return 1.0 if expected in predicted else 0.0

def calculate_accuracy(numb_of_predicted: float, total: float) -> float:
    """For calculate Accuracy"""
    if total:
        return (numb_of_predicted / total) * 100
    return 0

def calculate_cer(predicted: str, expected: str) -> float:
    """For calculate CER"""
    if not expected:
        return 0.0 if not predicted else 1.0

    distance = Levenshtein.distance(predicted, expected)
    cer = distance / len(expected)
    return cer

def calculate_acer(cer_lst: list) -> float:
    """For calculate ACER"""
    if len(cer_lst):
        return min(100.0,(sum(cer_lst) / len(cer_lst)) * 100)
    return 0

def calculate_robustness(accuracy_clean: float, accuracy_noisy: float) -> float:
    """For calculate Robustness"""
    if accuracy_clean == 0:
        return 0.0
    return (accuracy_noisy / accuracy_clean) * 100

class PerformanceMonitor:
    """For calculate time_inf (s), Memory_max (MB)"""
    def __init__(self):
        self.inference_times = []
        self.memory_usage = []
        self.process = psutil.Process()

    def start_measure(self):
        """Start timing"""
        self.start_time = time.perf_counter()
        self.start_memory = self._get_memory_mb()

    def end_measure(self):
        """Finish timing and record"""
        end_time = time.perf_counter()
        inference_time = end_time - self.start_time
        self.inference_times.append(inference_time)

        end_memory = self._get_memory_mb()
        self.memory_usage.append(end_memory)

        return inference_time

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / (1024 * 1024)

    def get_peak_memory_mb(self) -> float:
        """Peak memory consumption in MB"""
        return max(self.memory_usage) if self.memory_usage else 0.0

    def get_average_time(self) -> float:
        """Average inference time in seconds"""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)

    def get_fps(self) -> float:
        """FPS"""
        avg_time = self.get_average_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def reset(self):
        """Reset all measurements"""
        self.inference_times = []
        self.memory_usage = []

def get_model_size(model_path: str) -> float:
    """For calculate size_of_model (MB)"""
    cache_path = os.path.expanduser("~/.cache/huggingface/hub")

    total_size = 0

    if os.path.exists(cache_path):
        for root, dirs, files in os.walk(cache_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)

    size_mb = total_size / (1024 * 1024)
    # size_gb = size_mb / 1024

    return size_mb


def print_metrics_report(metrics: dict, model_name: str = "Model"):
    print(f"\n{'=' * 50}")
    print(f"Metrics report for {model_name}")
    print(f"{'=' * 50}")
    print(f"Total: {metrics['Total']} \n")

    print(f"Accuracy:   {metrics['Accuracy']:.2f}%")
    print(f"ACER:       {metrics['ACER']:.2f}%")
    print(f"Robustness: {metrics["Robustness"]:.2f}%\n")

    if metrics.get('Time_inf') is not None:
        print(f"Time inference: {metrics['Time_inf']:.3f} s")
        print(f"FPS:            {metrics['FPS']:.2f}")
        print(f"Memory max:     {metrics.get('Memory_max', 0):.2f} MB")

    print(f"Size of model:  {metrics['Size_of_model']:.2f} MB \n")

    print(f"{'=' * 50}\n")

