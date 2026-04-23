import os

# путь к папке с изображениями
folder = "C:/Users/artmn/ocr_gas_station/data/noisy"

# все файлы, начинающиеся с "stele_"
files = [f for f in os.listdir(folder) if f.startswith("stele_") and f.endswith(('.jpg', '.png', '.jpeg'))]

# # Отсортировать по имени (чтобы порядок был предсказуемым)
# files.sort()

# Перенумеровать
for i, filename in enumerate(files, 1):
    old_path = os.path.join(folder, filename)
    # Определить расширение
    ext = os.path.splitext(filename)[1]
    new_name = f"stele_{i+70}{ext}"  # 001, 002, 003...
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
    print(f"{filename} → {new_name}")

print(f"Готово! Переименовано {len(files)} файлов.")