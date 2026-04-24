import sys
import os
import re
import torch
import pandas as pd
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import (
    is_predicted, calculate_accuracy,
    calculate_cer, calculate_acer,
    calculate_robustness, PerformanceMonitor,
    print_metrics_report, get_model_size
)
from src.utils.load_data import load_images_and_labels

def extract_prices(text):
    return re.findall(r'\d+\.\d+', text)

def test_model_on_dataset(model, processor, images, all_prices_dicts, qualities, filenames):
    results = []
    cer_lst = []
    monitor = PerformanceMonitor()
    total_predictions = 0
    total_checks = 0

    for i in range(len(images)):
        monitor.start_measure()

        prompt = "Find all the gasoline prices at this gas station sign. Write only the numbers, separated by spaces, in the format XX.XX, where X is a digit."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=[text],
            images=[images[i]],
            padding=True,
            return_tensors="pt",
        )

        device = next(model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            predicted = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            predicted = predicted.replace(prompt, "").strip()

        monitor.end_measure()

        predicted_prices = extract_prices(predicted)
        prices_dict = all_prices_dicts[i]

        for fuel_type, expected_price in prices_dict.items():
            if expected_price in predicted_prices:
                pred_correct = 1.0
                cer = 0.0
            else:
                pred_correct = 0.0
                cer = 1.0

            total_predictions += pred_correct
            total_checks += 1
            cer_lst.append(cer)

            results.append({
                'filename': filenames[i],
                'fuel_type': fuel_type,
                'expected': expected_price,
                'predicted': predicted,
                'predicted_prices': predicted_prices,
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


def print_final_report(raw_metrics, noisy_metrics, model_size_mb, model_name="Qwen2.5-VL-3B"):
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


def main():
    print("Model is installing right now...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",trust_remote_code=True)
    print("Model is working\n")

    model_size_mb = get_model_size("Qwen/Qwen2.5-VL-3B-Instruct")

    r_images, r_expected_prices, r_qualities, r_filenames = load_images_and_labels('../data/raw', '../data/labels/labels_raw.csv')
    n_images, n_expected_prices, n_qualities, n_filenames = load_images_and_labels('../data/noisy', '../data/labels/labels_noisy.csv')

    raw_metrics = test_model_on_dataset(model, processor, r_images, r_expected_prices, r_qualities, r_filenames)
    noisy_metrics = test_model_on_dataset(model, processor, n_images, n_expected_prices, n_qualities, n_filenames)

    print_final_report(raw_metrics, noisy_metrics, model_size_mb, "Qwen2.5-VL-3B")

    df_results_raw = pd.DataFrame(raw_metrics['Results'])
    df_results_noisy = pd.DataFrame(noisy_metrics['Results'])
    df_results_raw.to_csv('../results/qwen_results_raw.csv', index=False)
    df_results_noisy.to_csv('../results/qwen_results_noisy.csv', index=False)

    print("\nResults saved:")
    print("  - results/qwen_results_raw.csv (raw)")
    print("  - results/qwen_results_noisy.csv (noisy)")


if __name__ == "__main__":
    main()