import os
import shutil
import pandas as pd


def copy_images(csv_path, source_folder, dest_folder):
    """Копирует изображения из source в dest по списку из CSV"""

    os.makedirs(dest_folder, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df['filename'] = df['filename'].str.strip()

    for filename in df['filename']:
        src = os.path.join(source_folder, filename)
        dst = os.path.join(dest_folder, filename)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Копировано: {filename}")
        else:
            print(f"Не найден: {src}")


# Копируем чистые изображения (НЕ трогаем data/raw)
copy_images(
    '../data/labels/labels_raw.csv',
    '../data/raw',  # исходная папка (не трогаем)
    '../data/prepared/raw_prepare'  # папка для обучения
)

# Копируем зашумленные изображения
copy_images(
    '../data/labels/labels_noisy.csv',
    '../data/noisy',  # исходная папка
    '../data/prepared/noisy_prepare'  # папка для обучения
)

print("\nГотово! Изображения скопированы в data/prepared/")