import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import easyocr
from PIL import Image

from src.utils.metrics import (
    is_predicted, calculate_accuracy,
    calculate_cer, calculate_acer,
    calculate_robustness, PerformanceMonitor,
    print_metrics_report
)
from src.utils.load_data import load_images_and_labels


def get_easyocr_size():
    easyocr_cache = os.path.expanduser("~/.EasyOCR/model")
    total = 0
    if os.path.exists(easyocr_cache):
        for root, dirs, files in os.walk(easyocr_cache):
            for file in files:
                total += os.path.getsize(os.path.join(root, file))
    return total / (1024 * 1024)


def test_model_on_dataset(reader, images, all_prices_dicts, qualities, filenames):
    results = []
    cer_lst = []
    monitor = PerformanceMonitor()
    total_predictions = 0
    total_checks = 0

    for i in range(len(images)):
        monitor.start_measure()

        img_np = np.array(images[i])
        result = reader.readtext(img_np, detail=0, paragraph=False)
        predicted = " ".join(result) if result else ""

        monitor.end_measure()

        prices_dict = all_prices_dicts[i]

        for fuel_type, expected_price in prices_dict.items():
            pred_correct = is_predicted(predicted, expected_price)
            total_predictions += pred_correct
            total_checks += 1

            cer = calculate_cer(predicted, expected_price)
            cer_lst.append(cer)

            results.append({
                'filename': filenames[i],
                'fuel_type': fuel_type,
                'expected': expected_price,
                'predicted': predicted[:100] if predicted else "",
                'correct': pred_correct,
                'cer': cer
            })

    accuracy = (total_predictions / total_checks) * 100 if total_checks > 0 else 0
    acer = calculate_acer(cer_lst)

    return {
        'Results': results,
        'Accuracy': accuracy,
        'ACER': acer,
        'Time_inf': monitor.get_average_time(),
        'FPS': monitor.get_fps(),
        'Memory_max': monitor.get_peak_memory_mb(),
        'Total': total_checks
    }


def print_final_report(raw_metrics, noisy_metrics, model_size_mb, model_name="EasyOCR"):
    robustness = calculate_robustness(raw_metrics['Accuracy'], noisy_metrics['Accuracy'])
    time_inf = (raw_metrics['Time_inf'] + noisy_metrics['Time_inf']) / 2
    fps = (raw_metrics['FPS'] + noisy_metrics['FPS']) / 2
    memory_max = max(raw_metrics['Memory_max'], noisy_metrics['Memory_max'])
    accuracy = (raw_metrics['Accuracy'] + noisy_metrics['Accuracy']) / 2
    acer = (raw_metrics['ACER'] + noisy_metrics['ACER']) / 2
    total = raw_metrics['Total'] + noisy_metrics['Total']

    metrics = {
        'Total': total,
        'Accuracy': accuracy,
        'ACER': acer,
        'Robustness': robustness,
        'Time_inf': time_inf,
        'FPS': fps,
        'Memory_max': memory_max,
        'Size_of_model': model_size_mb,
    }

    print_metrics_report(metrics, model_name)

def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Model is installing right now...")
    reader = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
    print("Model is working")

    r_images, r_expected_prices, r_qualities, r_filenames = load_images_and_labels('data/raw', 'data/labels/labels_raw.csv')
    n_images, n_expected_prices, n_qualities, n_filenames = load_images_and_labels('data/noisy', 'data/labels/labels_noisy.csv')

    print(f"Raw images: {len(r_images)}")
    print(f"Noisy images: {len(n_images)}\n")

    print("Testing on raw images...")
    raw_metrics = test_model_on_dataset(reader, r_images, r_expected_prices, r_qualities, r_filenames)
    print("Testing on noisy images...")
    noisy_metrics = test_model_on_dataset(reader, n_images, n_expected_prices, n_qualities, n_filenames)

    model_size_mb = get_easyocr_size()
    print_final_report(raw_metrics, noisy_metrics, model_size_mb, "EasyOCR")

    os.makedirs('results', exist_ok=True)
    pd.DataFrame(raw_metrics['Results']).to_csv('results/easyocr_raw.csv', index=False)
    pd.DataFrame(noisy_metrics['Results']).to_csv('results/easyocr_noisy.csv', index=False)

    print("\nResults saved:")
    print("  - results/easyocr_raw.csv")
    print("  - results/easyocr_noisy.csv")


if __name__ == "__main__":
    main()