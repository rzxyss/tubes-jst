import os
import random
import shutil

source_dir = "dataset"
output_dir = "output"

splits = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    files = os.listdir(class_path)
    random.shuffle(files)

    n = len(files)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split, split_list in split_files.items():
        split_class_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for file in split_list:
            shutil.copy(
                os.path.join(class_path, file),
                os.path.join(split_class_dir, file)
            )
