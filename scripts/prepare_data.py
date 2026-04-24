import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


def prepare_dataset(csv_path, images_folder, output_prefix):
    """Создаёт train/val/test JSON из CSV"""

    # Загрузить CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['filename'] = df['filename'].str.strip()

    # Создаём список всех образцов (изображение + текст)
    samples = []
    for _, row in df.iterrows():
        for col in ['price_92', 'price_95', 'price_98', 'price_DT']:
            price = row[col]
            if pd.notna(price):
                samples.append({
                    'file': os.path.join(images_folder, row['filename']),
                    'text': str(price)
                })

    # Разделяем на train (70%), val (15%), test (15%)
    train_samples, temp_samples = train_test_split(samples, test_size=0.3, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

    # Создаём папку для аннотаций
    os.makedirs('../data/prepared/annotations', exist_ok=True)

    # Сохраняем JSON
    with open(f'../data/prepared/annotations/{output_prefix}_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, indent=2, ensure_ascii=False)

    with open(f'../data/prepared/annotations/{output_prefix}_val.json', 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, indent=2, ensure_ascii=False)

    with open(f'../data/prepared/annotations/{output_prefix}_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)

    print(f"{output_prefix} - Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")


# Подготовка чистых изображений
prepare_dataset(
    '../data/labels/labels_raw.csv',
    '../data/prepared/raw_prepare',  # копия чистых изображений
    'raw'
)

# Подготовка зашумленных изображений
prepare_dataset(
    '../data/labels/labels_noisy.csv',
    '../data/prepared/noisy_prepare',  # копия шумных изображений
    'noisy'
)

print("\nГотово! JSON файлы созданы в data/prepared/annotations/")