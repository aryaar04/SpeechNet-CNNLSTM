import os
import zipfile

# === CONFIG ===
BASE_PATH = r"C:\Users\VIJAY\Downloads"
ZIP_FILES = [
    "malyalam_male_english.zip",
    "malyalam_female_english.zip"
]

# === UNZIP ===
for zip_name in ZIP_FILES:
    zip_path = os.path.join(BASE_PATH, zip_name)
    folder_name = zip_name.replace(".zip", "")
    target_folder = os.path.join(BASE_PATH, folder_name)

    print(f"\n[INFO] Checking: {zip_path}")

    if not os.path.exists(zip_path):
        print(f"[ERROR] File not found: {zip_path}")
        continue

    if not os.path.exists(target_folder):
        print(f"[ACTION] Extracting to {target_folder} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_PATH)
        print(f"[DONE] Extracted: {target_folder}")
    else:
        print(f"[SKIP] Folder already exists: {target_folder}")

print("\n✅ All datasets are ready.")
