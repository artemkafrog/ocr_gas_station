import sys
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'parseq'))

from src.utils.metrics import (
    is_predicted, calculate_accuracy,
    calculate_cer, calculate_acer,
    calculate_robustness, PerformanceMonitor,
    print_metrics_report
)
from src.utils.load_data import load_images_and_labels
from parseq.strhub.models.parseq.system import PARSeq

def test_model_on_dataset(model, images, all_prices_dicts, qualities, filenames):
    results = []
    cer_lst = []
    monitor = PerformanceMonitor()
    total_predictions = 0
    total_checks = 0

    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    for i in range(len(images)):
        monitor.start_measure()

        img_tensor = transform(images[i]).unsqueeze(0)
        with torch.no_grad():
            logits = model(img_tensor)
            pred = logits.softmax(-1)
            label, confidence = model.tokenizer.decode(pred)
            predicted = label[0]

        monitor.end_measure()
        prices_dict = all_prices_dicts[i]

        for fuel_type, expected_price in prices_dict.items():
            pred_correct = 1.0 if expected_price in predicted else 0.0
            total_predictions += pred_correct
            total_checks += 1

            cer = calculate_cer(predicted, expected_price)
            cer_lst.append(cer)

            results.append({
                'filename': filenames[i],
                'fuel_type': fuel_type,
                'expected': expected_price,
                'predicted': predicted,
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


def print_final_report(raw_metrics, noisy_metrics, model_size_mb, model_name="PARSeq"):
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
    return metrics


def get_parseq_model_size(ckpt_path: str = 'parseq_gas_model.ckpt') -> float:
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        return size_mb
    return 450.0


def main():
    print("Model is installing right now...")
    model = PARSeq.load_from_checkpoint('../src/models/parseq_gas_model.ckpt')
    model.eval()
    print("Model is working")

    r_images, r_expected_prices, r_qualities, r_filenames = load_images_and_labels('../data/raw', '../data/labels/labels_raw.csv')
    n_images, n_expected_prices, n_qualities, n_filenames = load_images_and_labels('../data/noisy', '../data/labels/labels_noisy.csv')

    model_size_mb = get_parseq_model_size('parseq_gas_model.ckpt')

    raw_metrics = test_model_on_dataset(model, r_images, r_expected_prices, r_qualities, r_filenames)
    noisy_metrics = test_model_on_dataset(model, n_images, n_expected_prices, n_qualities, n_filenames)

    print_final_report(raw_metrics, noisy_metrics, model_size_mb, "PARSeq (trained)")

    df_results_raw = pd.DataFrame(raw_metrics['Results'])
    df_results_noisy = pd.DataFrame(noisy_metrics['Results'])
    df_results_raw.to_csv('../results/parseq_trained_results_raw.csv', index=False)
    df_results_noisy.to_csv('../results/parseq_trained_results_noisy.csv', index=False)

    print("\nResults saved:")
    print("  - results/parseq_trained_results_raw.csv (raw)")
    print("  - results/parseq_trained_results_noisy.csv (noisy)")


if __name__ == "__main__":
    main()