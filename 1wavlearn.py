import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchcrepe
import numpy as np
from scipy.io.wavfile import write as wavwrite

# === 1. Загружаем сэмпл ===
sr = 16000  # torchcrepe требует 16 кГц
wav, sr0 = torchaudio.load("note.wav")
wav = torchaudio.functional.resample(wav, sr0, sr)
wav = torch.mean(wav, dim=0, keepdim=True)  # моно [1, T]

# === 2. Извлекаем f0 через torchcrepe ===
f0 = torchcrepe.predict(
    wav,
    sample_rate=sr,
    hop_length=160,       # 100 кадров в секунду
    fmin=50,
    fmax=2000,
    model="full",
    batch_size=2048,
    device="cpu",
    return_periodicity=False,
    pad=True,
)
f0 = f0.squeeze(0)  # [frames]

# === 3. Извлекаем громкость (RMS) ===
rms = torch.sqrt(torch.nn.functional.avg_pool1d(
    wav**2, kernel_size=160, stride=160
).squeeze(0) + 1e-8)

# === 4. Синхронизируем длины ===
min_len = min(f0.shape[0], rms.shape[0])
f0 = f0[:min_len]
rms = rms[:min_len]

x = torch.stack([torch.log1p(f0), torch.log1p(rms)], dim=-1)  # [frames,2]

# === 5. Модель ===
n_harm = 32
class HarmNet(nn.Module):
    def __init__(self, n_harm=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, n_harm+1)
        )
    def forward(self, x):
        out = self.net(x)
        harm = torch.nn.functional.softplus(out[...,:n_harm])
        noise = torch.nn.functional.softplus(out[...,-1:])
        return harm, noise

model = HarmNet(n_harm)
opt = optim.Adam(model.parameters(), lr=1e-3)

# === 6. Multi-Scale STFT Loss ===
def stft_loss(x, y):
    loss = 0
    for fft in [256, 512, 1024]:
        X = torch.stft(x, fft, hop_length=fft//4, return_complex=True)
        Y = torch.stft(y, fft, hop_length=fft//4, return_complex=True)
        loss += (X.abs() - Y.abs()).abs().mean()
    return loss

# === 7. Обучение ===
wave = wav.squeeze(0)  # [T]
T = wave.shape[0]
frames = x.shape[0]
hop = int(np.floor(T / frames))  # размер апсемплинга

t = torch.arange(T) / sr

for ep in range(201):
    harm, noise = model(x)   # [frames, n_harm], [frames,1]

    # Апсемплим параметры
    f0_up = torch.repeat_interleave(f0, hop)
    harm_up = torch.repeat_interleave(harm, hop, dim=0)
    noise_up = torch.repeat_interleave(noise[:,0], hop)

    # --- ВЫРАВНИВАНИЕ ДЛИН ---
    min_T = min(wave.shape[0], f0_up.shape[0], harm_up.shape[0], noise_up.shape[0])
    wave = wave[:min_T]
    t = torch.arange(min_T) / sr
    f0_up = f0_up[:min_T]
    harm_up = harm_up[:min_T]
    noise_up = noise_up[:min_T]

    # Синтез
    synth = torch.zeros(min_T)
    for k in range(1, n_harm+1):
        amp = harm_up[:,k-1]
        synth += amp * torch.sin(2*np.pi*k*f0_up*t)

    synth += torch.randn_like(synth) * noise_up * 0.01

    # Лосс
    loss = stft_loss(synth, wave)
    opt.zero_grad(); loss.backward(); opt.step()

    if ep % 20 == 0:
        print(f"ep {ep}, loss {float(loss):.4f}")

# === 8. Сохраняем результат ===
synth = synth.detach().numpy()
wavwrite("note_synth.wav", sr, (synth*32767).astype(np.int16))
print("Сохранён файл note_synth.wav")
