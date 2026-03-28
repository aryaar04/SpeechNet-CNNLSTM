import os
import re
import random
import json
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import wer
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ================================
# CONFIG
# ================================
MANIFEST = "manifest.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_MELS = 80
FFT = 400
HOP = 160
BATCH_SIZE = 8
NUM_EPOCHS =50   # run 50 more epochs 
LEARNING_RATE = 1e-4
PATIENCE = 10
OUTPUT_DIR = "cnn_lstm_augmented_model"
SEED = 42
NUM_WORKERS = 0
LOG_CSV = os.path.join(OUTPUT_DIR, "training_log.csv")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pt")

# ================================
# REPRODUCIBILITY
# ================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# ================================
# TEXT TOKENIZER
# ================================
def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z' ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class CharTokenizer:
    def __init__(self, transcripts):
        alltext = " ".join(transcripts)
        chars = sorted(list(set(alltext)))
        if " " not in chars:
            chars.append(" ")
        self.pad_token = "<pad>"
        self.blank_token = "<ctc_blank>"
        tokens = [self.pad_token, self.blank_token] + chars
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.blank_id = self.token_to_id[self.blank_token]
        self.pad_id = self.token_to_id[self.pad_token]
        self.vocab_size = len(tokens)

    def encode(self, text):
        return [self.token_to_id[c] for c in text]

    def decode(self, ids):
        toks = []
        for i in ids:
            if i in (self.blank_id, self.pad_id):
                continue
            toks.append(self.id_to_token.get(int(i), ""))
        return "".join(toks)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

# ================================
# DATASET
# ================================
class ASRDataset(Dataset):
    def __init__(self, manifest_path, tokenizer, augment=False):
        df = pd.read_csv(manifest_path)
        df["transcript"] = df["transcript"].astype(str).map(clean_text)
        df = df[df["transcript"].str.len() > 0].reset_index(drop=True)
        self.df = df
        self.tokenizer = tokenizer
        self.augment = augment

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=FFT, hop_length=HOP, n_mels=N_MELS
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_aug = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def add_noise(self, waveform):
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path, text = row["wav_path"], row["transcript"]
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)
        if self.augment and random.random() < 0.3:
            waveform = self.add_noise(waveform)
        mel = self.mel_spec(waveform)
        mel_db = self.to_db(mel)
        if self.augment and random.random() < 0.5:
            mel_db = self.spec_aug(mel_db)
            mel_db = self.time_aug(mel_db)
        mel_db = mel_db.squeeze(0).T
        label_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return mel_db, label_ids, wav_path, text

    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    mels, labels, paths, texts = zip(*batch)
    lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    padded_mels = pad_sequence(mels, batch_first=True).permute(0, 2, 1).unsqueeze(1)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return padded_mels, lengths, padded_labels, label_lengths, paths, texts

# ================================
# MODEL
# ================================
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, n_mels=N_MELS, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.project = nn.Linear(128 * n_mels, hidden)
    def forward(self, x):
        c = self.conv(x)
        B, C, H, W = c.shape
        c = c.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        return self.project(c)

class CNNLSTM_ASR(nn.Module):
    def __init__(self, vocab_size, enc_hidden=256, lstm_hidden=512, lstm_layers=2):
        super().__init__()
        self.encoder = ConvEncoder(hidden=enc_hidden)
        self.lstm = nn.LSTM(enc_hidden, lstm_hidden // 2, num_layers=lstm_layers,
                            bidirectional=True, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(lstm_hidden, vocab_size)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self, x):
        enc = self.encoder(x)
        out, _ = self.lstm(enc)
        logits = self.classifier(out)
        return self.log_softmax(logits).permute(1, 0, 2)

# ================================
# TRAINING FUNCTIONS
# ================================
def pack_targets(padded_labels, label_lengths, device):
    targets = [padded_labels[i, :l].to(torch.int64) for i, l in enumerate(label_lengths.tolist()) if l > 0]
    return torch.cat(targets).to(device)

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train"):
        mels, _, padded_labels, label_lens, _, _ = batch
        mels, padded_labels, label_lens = mels.to(device), padded_labels.to(device), label_lens.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            log_probs = model(mels)
            T = log_probs.size(0)
            input_lens = torch.full((log_probs.size(1),), T, dtype=torch.long, device=device)
            targets = pack_targets(padded_labels, label_lens, device)
            loss = criterion(log_probs, targets, input_lens, label_lens)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss, preds, refs = 0.0, [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            mels, _, padded_labels, label_lens, _, texts = batch
            mels, padded_labels, label_lens = mels.to(device), padded_labels.to(device), label_lens.to(device)
            log_probs = model(mels)
            T = log_probs.size(0)
            input_lens = torch.full((log_probs.size(1),), T, dtype=torch.long, device=device)
            targets = pack_targets(padded_labels, label_lens, device)
            loss = criterion(log_probs, targets, input_lens, label_lens)
            total_loss += loss.item()
            preds_batch = log_probs.permute(1, 0, 2).argmax(-1).cpu().numpy()
            for i, p in enumerate(preds_batch):
                prev, out = None, []
                for x in p:
                    if x == tokenizer.blank_id:
                        prev = None
                        continue
                    if x != prev:
                        out.append(x)
                    prev = x
                preds.append(tokenizer.decode(out))
                refs.append(texts[i])
    return total_loss / len(dataloader), wer(refs, preds)

# ================================
# MAIN TRAINING LOOP
# ================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(MANIFEST)
    df["transcript"] = df["transcript"].astype(str).map(clean_text)
    tokenizer = CharTokenizer(df["transcript"].tolist())
    tokenizer.save(OUTPUT_DIR)

    train_csv = os.path.join(OUTPUT_DIR, "train.csv")
    val_csv = os.path.join(OUTPUT_DIR, "val.csv")
    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        n_val = max(1, int(0.1 * len(df)))
        df.iloc[n_val:].to_csv(train_csv, index=False)
        df.iloc[:n_val].to_csv(val_csv, index=False)

    train_ds = ASRDataset(train_csv, tokenizer, augment=True)
    val_ds = ASRDataset(val_csv, tokenizer, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CNNLSTM_ASR(vocab_size=tokenizer.vocab_size).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=True)

    try:
        scaler = torch.amp.GradScaler(device_type="cuda" if DEVICE == "cuda" else "cpu")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_wer = 1.0
    start_epoch = 1

    # --- Resume from checkpoint if available ---
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt.get("optimizer_state", optimizer.state_dict()))
        best_wer = ckpt.get("best_wer", 1.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"[RESUME] Found checkpoint at epoch {ckpt['epoch']}, starting from epoch {start_epoch}.")
    elif os.path.exists(LOG_CSV):
        try:
            df_log = pd.read_csv(LOG_CSV)
            if not df_log.empty:
                last_epoch = int(df_log["epoch"].iloc[-1])
                start_epoch = last_epoch + 1
                print(f"[AUTO-RESUME] No checkpoint found. Starting from epoch {start_epoch} (based on log).")
        except Exception:
            pass

    # --- Logging header ---
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w") as f:
            f.write("epoch,train_loss,val_loss,val_wer\n")

    # --- Main training loop ---
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        print(f"\n[Epoch {epoch}] --------------------------")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_wer = evaluate(model, val_loader, criterion, tokenizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WER: {val_wer*100:.2f}%")

        scheduler.step(val_loss)
        with open(LOG_CSV, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_wer:.6f}\n")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_wer": best_wer
        }
        torch.save(ckpt, CHECKPOINT_PATH)

        if val_wer < best_wer:
            best_wer = val_wer
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"[SAVED] New best model (WER {best_wer*100:.2f}%)")

        if os.path.exists(LOG_CSV):
            df_log = pd.read_csv(LOG_CSV)
            print("\n📈 Last 5 Epochs Summary:")
            print(df_log.tail(5).to_string(index=False))

    print("\n✅ Training Complete. Best Val WER:", f"{best_wer*100:.2f}%")

if __name__ == "__main__":
    main()
