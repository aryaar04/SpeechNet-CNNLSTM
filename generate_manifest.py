import os
import csv
import re

# === CONFIG ===
DATASETS = [
    r"C:\Users\VIJAY\Downloads\malyalam_male_english",
    r"C:\Users\VIJAY\Downloads\malyalam_female_english"
]
OUTPUT_FILE = "manifest.csv"

pattern = re.compile(r'\(\s*(\S+)\s+"(.+?)"\s*\)')

def read_lines_safely(path):
    """Reads text file robustly with fallback encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "windows-1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to read {path} with all fallback encodings")

rows = []
for dataset_dir in DATASETS:
    txt_path = os.path.join(dataset_dir, "txt.done.data")
    wav_dir = os.path.join(dataset_dir, "wav")  # 🔥 FIXED: correct folder name

    if not os.path.exists(txt_path):
        print(f"[WARN] Missing txt.done.data in {dataset_dir}")
        continue

    if not os.path.exists(wav_dir):
        print(f"[ERROR] WAV folder not found: {wav_dir}")
        continue

    lines = read_lines_safely(txt_path)

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            wav_id, transcript = match.groups()
            wav_file = os.path.join(wav_dir, f"{wav_id}.wav")
            if os.path.exists(wav_file):
                rows.append([wav_file, transcript.strip().lower()])
            else:
                print(f"[MISSING WAV] {wav_file}")

print(f"[INFO] Found {len(rows)} audio-transcript pairs")

# Write to CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["wav_path", "transcript"])
    writer.writerows(rows)

print(f"[DONE] Manifest saved as {OUTPUT_FILE}")
