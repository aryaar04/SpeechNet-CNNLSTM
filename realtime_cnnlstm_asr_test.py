import os
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from cnn_lstm_asr_augmented import CNNLSTM_ASR, CharTokenizer, clean_text, SAMPLE_RATE, N_MELS, FFT, HOP, OUTPUT_DIR, DEVICE

# ================================
# CONFIG
# ================================
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
RECORD_SECONDS = 5  # duration of audio capture
SAVE_WAV = False    # set True to save recorded audio

# ================================
# LOAD TOKENIZER
# ================================
def load_tokenizer_from_manifest(manifest="manifest.csv"):
    import pandas as pd
    df = pd.read_csv(manifest)
    df["transcript"] = df["transcript"].astype(str).map(clean_text)
    transcripts = df["transcript"].tolist()
    return CharTokenizer(transcripts)

# ================================
# LOAD MODEL
# ================================
def load_model(model_path, tokenizer):
    model = CNNLSTM_ASR(vocab_size=tokenizer.vocab_size).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# ================================
# RECORD AUDIO
# ================================
def record_audio(duration=RECORD_SECONDS):
    print(f"\n🎙️ Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")
    waveform = torch.tensor(audio.T, dtype=torch.float32)
    return waveform

# ================================
# PREPROCESS AUDIO
# ================================
def preprocess_audio(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=FFT,
        hop_length=HOP,
        n_mels=N_MELS
    )
    to_db = torchaudio.transforms.AmplitudeToDB()
    mel_db = to_db(mel_spec(waveform))
    mel_db = mel_db.squeeze(0).T  # (time, n_mels)
    return mel_db

# ================================
# GREEDY DECODING
# ================================
def greedy_decode(log_probs, tokenizer):
    probs = log_probs.permute(1, 0, 2).argmax(-1).cpu().numpy()
    preds = []
    for p in probs:
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
    return preds[0] if preds else ""

# ================================
# MAIN LOOP
# ================================
def main():
    print("[INFO] Loading tokenizer and model...")
    tokenizer = load_tokenizer_from_manifest("manifest.csv")
    model = load_model(MODEL_PATH, tokenizer)

    while True:
        cmd = input("\nPress ENTER to record or type 'exit' to quit: ").strip().lower()
        if cmd == "exit":
            break

        waveform = record_audio()
        if SAVE_WAV:
            torchaudio.save("input_audio.wav", waveform, SAMPLE_RATE)

        mel = preprocess_audio(waveform)
        mel = pad_sequence([mel], batch_first=True)
        mel = mel.permute(0, 2, 1).unsqueeze(1).to(DEVICE)

        with torch.no_grad():
            log_probs = model(mel)
            text = greedy_decode(log_probs, tokenizer)

        print(f"\n🗣️  Recognized Text: {text.upper()}")

if __name__ == "__main__":
    main()
