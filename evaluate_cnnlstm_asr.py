import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jiwer import wer, cer, process_words
from difflib import ndiff
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

# ===== IMPORT YOUR OWN MODULES =====
from cnn_lstm_asr_augmented import ASRDataset, CharTokenizer, collate_fn, CNNLSTM_ASR, clean_text, OUTPUT_DIR, DEVICE, SAMPLE_RATE, BATCH_SIZE

# ==============================
# CONFIG
# ==============================
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
MANIFEST_PATH = "manifest.csv"
RESULTS_DIR = "./evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# UTILS
# ==============================
def decode_prediction(log_probs, tokenizer):
    preds = []
    probs = log_probs.permute(1, 0, 2).argmax(-1).cpu().numpy()
    for i, p in enumerate(probs):
        prev = None
        out = []
        for x in p:
            if x == tokenizer.blank_id:
                prev = None
                continue
            if x != prev:
                out.append(x)
            prev = x
        preds.append(tokenizer.decode(out))
    return preds


def diff_strings(ref, pred):
    diff = ndiff(ref, pred)
    out = []
    for d in diff:
        if d.startswith("- "):
            out.append(f"-{d[2]}")
        elif d.startswith("+ "):
            out.append(f"+{d[2]}")
    return " ".join(out)


# ==============================
# LOAD EVERYTHING
# ==============================
print(f"[INFO] Loading model from {MODEL_PATH}")
df = pd.read_csv(MANIFEST_PATH)
df["transcript"] = df["transcript"].astype(str).map(clean_text)
tokenizer = CharTokenizer(df["transcript"].tolist())

dataset = ASRDataset(MANIFEST_PATH, tokenizer, augment=False)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

model = CNNLSTM_ASR(vocab_size=tokenizer.vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

criterion = torch.nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

# ==============================
# EVALUATION LOOP
# ==============================
total_loss = 0.0
pred_texts, ref_texts = [], []
sample_results = []

print("[INFO] Starting evaluation...")
with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating"):
        mels, _, labels, label_lens, paths, texts = batch
        mels, labels = mels.to(DEVICE), labels.to(DEVICE)

        log_probs = model(mels)
        T = log_probs.size(0)
        input_lens = torch.full((log_probs.size(1),), T, dtype=torch.long).to(DEVICE)
        loss = criterion(log_probs, labels, input_lens, label_lens.to(DEVICE))
        total_loss += loss.item()

        preds = decode_prediction(log_probs, tokenizer)
        pred_texts.extend(preds)
        ref_texts.extend(texts)

# ==============================
# METRICS
# ==============================
wer_score = wer(ref_texts, pred_texts)
cer_score = cer(ref_texts, pred_texts)
acc = (1 - wer_score) * 100
avg_loss = total_loss / len(loader)

print("\n===== EVALUATION RESULTS =====")
print(f"✅ Word Error Rate (WER): {wer_score * 100:.2f}%")
print(f"✅ Character Error Rate (CER): {cer_score * 100:.2f}%")
print(f"✅ Word-level Accuracy: {acc:.2f}%")
print(f"✅ Average Evaluation Loss: {avg_loss:.4f}\n")

# ==============================
# QUALITATIVE ANALYSIS
# ==============================
print("🔍 Running qualitative sample comparison...\n")
for i in range(min(10, len(ref_texts))):
    ref = ref_texts[i].upper()
    pred = pred_texts[i].upper() if pred_texts[i] else "<EMPTY>"
    sample_wer = wer([ref], [pred]) * 100
    sample_cer = cer([ref], [pred]) * 100
    diff = diff_strings(ref, pred)
    print(f"🎧 Sample {i+1}")
    print(f"🗣️  Ref : {ref}")
    print(f"🤖 Pred: {pred}")
    print(f"📊  WER: {sample_wer:.2f}% | CER: {sample_cer:.2f}%")
    print(f"🔎 Diff: {diff}\n")

# ==============================
# CONFUSION MATRIX (length-safe)
# ==============================
print("[INFO] Generating confusion matrix...")
all_chars = sorted(list(set("".join(ref_texts + pred_texts))))

y_true = []
y_pred = []
for ref, pred in zip(ref_texts, pred_texts):
    # pad/truncate so both are equal length
    min_len = min(len(ref), len(pred))
    ref_trim = list(ref[:min_len])
    pred_trim = list(pred[:min_len])
    y_true.extend(ref_trim)
    y_pred.extend(pred_trim)

cm = confusion_matrix(y_true, y_pred, labels=all_chars)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 10))
sns.heatmap(
    np.log1p(cm),  # log scale makes it more readable
    cmap="viridis",
    xticklabels=all_chars,
    yticklabels=all_chars,
    cbar_kws={'label': 'Log(Frequency)'},
)
plt.title("Character-Level Confusion Matrix (Heatmap View)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_heatmap.png"), dpi=300)
plt.close()

