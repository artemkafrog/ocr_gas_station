import pandas as pd
import os
from PIL import Image

def load_images_and_labels(images_folder, labels_file):
    df = pd.read_csv(labels_file)

    images = []
    expected_prices = []
    qualities = []
    filenames = []

    df.columns = df.columns.str.strip()
    df['filename'] = df['filename'].str.strip()
    price_columns = [col for col in df.columns if col.startswith('price_')]

    missing_files = []  # список пропущенных

    for _, row in df.iterrows():
        img_path = os.path.join(images_folder, row['filename'])
        if os.path.exists(img_path):
            images.append(Image.open(img_path).convert('RGB'))
            prices_dict = {}
            for col in price_columns:
                price = row[col]
                if pd.notna(price):
                    prices_dict[col] = str(price)

            expected_prices.append(prices_dict)
            qualities.append(row.get('quality', 'unknown'))
            filenames.append(row['filename'])

    print(f"Загружено: {len(images)} изображений")
    print(f"Пропущено (файлы не найдены): {len(missing_files)}")
    if missing_files:
        print("Примеры пропущенных файлов:", missing_files[:5])

    return images, expected_prices, qualities, filenames

def load_all_prices(labels_file):
    df = pd.read_csv(labels_file)
    return df