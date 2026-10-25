import os

# Replace this with the actual path of your color folder
DATASET_PATH = r"C:\Users\cw\Desktop\Disease detection\color"

# Check whether the dataset folder exists
if not os.path.isdir(DATASET_PATH):
    print("ERROR: Dataset folder not found.")
    print("Please check your dataset path.")
else:
    print("Dataset folder found!")

    # Get all subfolders containing disease classes
    folders = [
        name for name in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, name))
    ]

    print("Number of class folders:", len(folders))
    print("\nFirst 10 classes:")

    for folder in folders[:10]:
        folder_path = os.path.join(DATASET_PATH, folder)

        # Count image files in each class folder
        images = [
            name for name in os.listdir(folder_path)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"{folder}: {len(images)} images")
