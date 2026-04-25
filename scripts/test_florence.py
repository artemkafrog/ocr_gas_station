import sys
import types
import importlib.machinery

# --- need to fix flash_attn---
flash_attn = types.ModuleType("flash_attn")
flash_attn.__spec__ = importlib.machinery.ModuleSpec(name="flash_attn", loader=None)
flash_attn_interface = types.ModuleType("flash_attn.flash_attn_interface")
flash_attn_interface.__spec__ = importlib.machinery.ModuleSpec(name="flash_attn.flash_attn_interface", loader=None)

def dummy_func(*args, **kwargs):
    raise NotImplementedError("flash_attn is not available on this system")

flash_attn_interface.flash_attn_func = dummy_func
flash_attn_interface.flash_attn_varlen_func = dummy_func

flash_attn.flash_attn_interface = flash_attn_interface

sys.modules["flash_attn"] = flash_attn
sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface

import sys
import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.metrics import (
    calculate_cer, calculate_acer,
    calculate_robustness, PerformanceMonitor,
    get_model_size, print_metrics_report
)
from src.utils.load_data import load_images_and_labels


def test_model_on_dataset(processor, model, images, all_prices_dicts, qualities, filenames):
    results = []
    cer_lst = []
    monitor = PerformanceMonitor()
    total_predictions = 0
    total_checks = 0

    prompt = "<OCR>"
    device = next(model.parameters()).device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    for i in range(len(images)):
        monitor.start_measure()

        image = images[i].convert("RGB")

        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch_dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values", None),
                max_new_tokens=50,
                do_sample=False
            )

        predicted = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

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

        print(f"{filenames[i]}: {predicted}")

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


def print_final_report(raw_metrics, noisy_metrics, model_size_mb, model_name="Florence-2-base"):
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
    print("Model is installing right now...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base",trust_remote_code=True)
    print("GPU:" if torch.cuda.is_available() else "CPU:", device)
    print("Model is working")

    r_images, r_expected_prices, r_qualities, r_filenames = load_images_and_labels('../data/raw', '../data/labels/labels_raw.csv')
    n_images, n_expected_prices, n_qualities, n_filenames = load_images_and_labels('../data/noisy', '../data/labels/labels_noisy.csv')

    model_size_mb = get_model_size("microsoft/Florence-2-base")

    print("Testing on raw images...")
    raw_metrics = test_model_on_dataset( processor, model, r_images, r_expected_prices, r_qualities, r_filenames)
    print("Testing on noisy images...")
    noisy_metrics = test_model_on_dataset(processor, model, n_images, n_expected_prices, n_qualities, n_filenames)

    print_final_report(raw_metrics, noisy_metrics, model_size_mb)

    os.makedirs('../results', exist_ok=True)
    pd.DataFrame(raw_metrics['Results']).to_csv('../results/florence_base_results_raw.csv', index=False)
    pd.DataFrame(noisy_metrics['Results']).to_csv('../results/florence_base_results_noisy.csv', index=False)

    print("\nResults saved:")
    print("  - results/florence_base_results_raw.csv")
    print("  - results/florence_base_results_noisy.csv")


if __name__ == "__main__":
    main()