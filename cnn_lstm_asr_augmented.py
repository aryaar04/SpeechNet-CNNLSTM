# cnn_lstm_asr_augmented.py
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
NUM_EPOCHS = 25
LEARNING_RATE = 3e-4
PATIENCE = 3
OUTPUT_DIR = "cnn_lstm_augmented_model"
SEED = 42
NUM_WORKERS = 0  # keep 0 on Windows
LOG_CSV = os.path.join(OUTPUT_DIR, "training_log.csv")

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
        # token -> id
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        # id -> token
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.blank_id = self.token_to_id[self.blank_token]
        self.pad_id = self.token_to_id[self.pad_token]
        self.vocab_size = len(tokens)

    def encode(self, text):
        # returns list[int]
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
# DATASET WITH AUGMENTATION
# ================================
class ASRDataset(Dataset):
    def __init__(self, manifest_path, tokenizer, augment=False):
        df = pd.read_csv(manifest_path)
        df["transcript"] = df["transcript"].astype(str).map(clean_text)
        df = df[df["transcript"].str.len() > 0].reset_index(drop=True)
        self.df = df
        self.tokenizer = tokenizer
        self.augment = augment

        # transforms
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=FFT,
            hop_length=HOP,
            n_mels=N_MELS
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_aug = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_aug = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def add_noise(self, waveform):
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["wav_path"]
        text = row["transcript"]

        # load audio (torchaudio will fallback to soundfile if built that way)
        waveform, sr = torchaudio.load(wav_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)  # to mono

        if self.augment and random.random() < 0.3:
            waveform = self.add_noise(waveform)

        mel = self.mel_spec(waveform)      # (1, n_mels, time_frames)
        mel_db = self.to_db(mel)

        if self.augment and random.random() < 0.5:
            mel_db = self.spec_aug(mel_db)
            mel_db = self.time_aug(mel_db)

        # output shape for model: (time, n_mels)
        mel_db = mel_db.squeeze(0).T  # (time, n_mels)
        label_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        return mel_db, label_ids, wav_path, text

    def __len__(self):
        return len(self.df)

# ================================
# COLLATE FUNCTION
# ================================
def collate_fn(batch):
    mels, labels, paths, texts = zip(*batch)
    lengths = torch.tensor([m.shape[0] for m in mels], dtype=torch.long)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    padded_mels = pad_sequence(mels, batch_first=True)               # (B, T_max, n_mels)
    padded_mels = padded_mels.permute(0, 2, 1).unsqueeze(1)          # (B, 1, n_mels, T_max) -> conv expects (B,C,H,W)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)

    return padded_mels, lengths, padded_labels, label_lengths, paths, texts

# ================================
# MODEL
# ================================
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, n_mels=N_MELS, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.project = nn.Linear(128 * n_mels, hidden)

    def forward(self, x):
        # x: (B, 1, n_mels, T)
        c = self.conv(x)                         # (B, C, n_mels, T)
        B, C, H, W = c.shape
        c = c.permute(0, 3, 1, 2).contiguous()   # (B, T, C, H)
        c = c.view(B, W, C * H)                  # (B, T, C*H)
        return self.project(c)                   # (B, T, hidden)

class CNNLSTM_ASR(nn.Module):
    def __init__(self, vocab_size, enc_hidden=256, lstm_hidden=512, lstm_layers=2):
        super().__init__()
        self.encoder = ConvEncoder(hidden=enc_hidden)
        self.lstm = nn.LSTM(enc_hidden, lstm_hidden // 2, num_layers=lstm_layers,
                            bidirectional=True, batch_first=True, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(lstm_hidden, vocab_size)
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: (B,1,n_mels,T)
        enc = self.encoder(x)            # (B, T, enc_hidden)
        out, _ = self.lstm(enc)          # (B, T, lstm_hidden)
        logits = self.classifier(out)    # (B, T, V)
        log_probs = self.log_softmax(logits)
        return log_probs.permute(1, 0, 2)  # (T, B, V) expected by CTCLoss

# ================================
# HELPERS FOR CTC
# ================================
def pack_targets(padded_labels: torch.Tensor, label_lengths: torch.Tensor, device):
    """
    padded_labels: (B, L) padded with pad_id
    label_lengths: (B,)
    returns:
      targets (1D int64): concatenated targets across batch
    """
    targets = []
    for i, l in enumerate(label_lengths.tolist()):
        if l > 0:
            targets.append(padded_labels[i, :l].to(torch.int64))
    if len(targets) == 0:
        return torch.tensor([], dtype=torch.int64, device=device)
    return torch.cat(targets).to(device)

# ================================
# TRAIN/EVAL
# ================================
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(dataloader, desc="train"):
        mels, _, padded_labels, label_lens, _, _ = batch
        mels = mels.to(device)
        padded_labels = padded_labels.to(device)
        label_lens = label_lens.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            log_probs = model(mels)                     # (T, B, V)
            T = log_probs.size(0)
            input_lens = torch.full((log_probs.size(1),), T, dtype=torch.long, device=device)
            targets = pack_targets(padded_labels, label_lens, device)  # 1D
            # CTCLoss signature: (log_probs, targets, input_lengths, target_lengths)
            loss = criterion(log_probs, targets, input_lens, label_lens)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)

def evaluate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0.0
    preds, refs = [], []
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval"):
            mels, _, padded_labels, label_lens, _, texts = batch
            mels = mels.to(device)
            padded_labels = padded_labels.to(device)
            label_lens = label_lens.to(device)

            log_probs = model(mels)               # (T, B, V)
            T = log_probs.size(0)
            input_lens = torch.full((log_probs.size(1),), T, dtype=torch.long, device=device)
            targets = pack_targets(padded_labels, label_lens, device)
            loss = criterion(log_probs, targets, input_lens, label_lens)
            total_loss += loss.item()
            n_batches += 1

            # Greedy decode
            preds_batch = log_probs.permute(1, 0, 2).argmax(-1).cpu().numpy()  # (B, T)
            for i, p in enumerate(preds_batch):
                prev = None
                out = []
                for x in p:
                    if int(x) == tokenizer.blank_id:
                        prev = None
                        continue
                    if x != prev:
                        out.append(int(x))
                    prev = int(x)
                preds.append(tokenizer.decode(out))
                refs.append(texts[i])
    wer_score = wer(refs, preds) if len(refs) > 0 else 1.0
    return (total_loss / max(1, n_batches)), wer_score, refs, preds

# ================================
# MAIN
# ================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(MANIFEST)
    df["transcript"] = df["transcript"].astype(str).map(clean_text)
    transcripts = df["transcript"].tolist()
    tokenizer = CharTokenizer(transcripts)
    tokenizer.save(OUTPUT_DIR)

    # split
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_val = max(1, int(0.1 * len(df)))
    val_df = df.iloc[:n_val]
    train_df = df.iloc[n_val:]
    train_csv = os.path.join(OUTPUT_DIR, "train.csv")
    val_csv = os.path.join(OUTPUT_DIR, "val.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_ds = ASRDataset(train_csv, tokenizer, augment=True)
    val_ds = ASRDataset(val_csv, tokenizer, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=NUM_WORKERS)

    model = CNNLSTM_ASR(vocab_size=tokenizer.vocab_size).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
    # GradScaler: new style
    # Version-safe GradScaler initialization
    try:
        scaler = torch.amp.GradScaler(device_type="cuda" if DEVICE == "cuda" else "cpu")
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # quick sanity check on single batch
    print("[SANITY] grabbing one batch for shape checks...")
    batch = next(iter(train_loader))
    mels, _, padded_labels, label_lens, _, _ = batch
    print(" mels.shape:", mels.shape, "padded_labels.shape:", padded_labels.shape, "label_lens:", label_lens[:5].tolist())
    # forward pass check
    with torch.no_grad():
        _ = model(mels.to(DEVICE))
    print("[SANITY] forward pass OK (no exceptions).")

    best_wer = 1.0
    patience = 0

    # init CSV logging
    with open(LOG_CSV, "w", encoding="utf-8") as logf:
        logf.write("epoch,train_loss,val_loss,val_wer\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_wer, refs, preds = evaluate(model, val_loader, criterion, tokenizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val WER: {val_wer*100:.2f}%")

        # append CSV
        with open(LOG_CSV, "a", encoding="utf-8") as logf:
            logf.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_wer:.6f}\n")

        # save qualitative samples for diagnostics
        sample_path = os.path.join(OUTPUT_DIR, f"eval_samples_epoch{epoch}.csv")
        if len(refs) > 0:
            pd.DataFrame({"ref": refs[:200], "pred": preds[:200]}).to_csv(sample_path, index=False)

        if val_wer < best_wer:
            best_wer = val_wer
            patience = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            # also save final tokenizer
            tokenizer.save(OUTPUT_DIR)
            print(f"[SAVED] Best model (WER {best_wer*100:.2f}%) -> {os.path.join(OUTPUT_DIR, 'best_model.pt')}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("[EARLY STOP] No improvement.")
                break

    print("\n✅ Training complete. Best Val WER:", f"{best_wer*100:.2f}%")

if __name__ == "__main__":
    main()
