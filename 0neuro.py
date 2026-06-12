# neural_synth.py
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from scipy.io.wavfile import write as wavwrite
from pathlib import Path
import librosa

sr = 22050
dur = 0.8
n_harm = 16
device = "cpu"

def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69.0)/12.0))

def adsr(t, a=0.01, d=0.08, s=0.7, r=0.15):
    env = np.zeros_like(t)
    a_len, d_len, r_len = int(a*sr), int(d*sr), int(r*sr)
    s_len = len(t) - a_len - d_len - r_len
    if s_len < 0:
        return np.exp(-5 * t / (t[-1]+1e-8))
    if a_len>0: env[:a_len] = np.linspace(0,1,a_len,endpoint=False)
    if d_len>0: env[a_len:a_len+d_len] = np.linspace(1,s,d_len,endpoint=False)
    if s_len>0: env[a_len+d_len:a_len+d_len+s_len] = s
    if r_len>0:
        start = a_len+d_len+s_len
        env[start:] = np.linspace(s,0,r_len,endpoint=True)
    return env

def extract_harmonics_from_wav(wav_path, f0_hz, n_harm=16, sr=22050):
    y, sr = librosa.load(wav_path, sr=sr)
    N = len(y)
    Y = np.fft.rfft(y * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1/sr)

    amps = np.zeros(n_harm, dtype=np.float32)
    for k in range(1, n_harm+1):
        # частота гармоники
        f_h = f0_hz * k
        # ищем ближайший бин FFT
        idx = np.argmin(np.abs(freqs - f_h))
        amps[k-1] = np.abs(Y[idx])

    # нормализация
    amps /= amps.max() + 1e-8

    # шум = остаток энергии, не объяснённый гармониками
    harmonic_energy = np.sum(amps**2)
    total_energy = np.sum(np.abs(Y)**2)
    noise = max(0.0, (total_energy - harmonic_energy) / (total_energy+1e-8))

    return amps, float(noise)


def teacher_params(instr, f0_hz, vel, wav_dir="samples"):
    """
    instr: имя инструмента (например "bass")
    f0_hz: частота ноты
    vel:   громкость (пока можно игнорировать, или использовать для масштабирования)
    """
    midi = 69 + 12*np.log2(f0_hz/440.0)
    midi = int(round(midi))

    wav_path = Path(wav_dir)/f"{instr}_midi{midi}.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Нет файла {wav_path}")

    amps, noise = extract_harmonics_from_wav(wav_path, f0_hz, n_harm)

    # ADSR пока можно зашить вручную или извлекать огибающую RMS
    env = (0.01, 0.1, 0.7, 0.2)

    # подстраиваем под velocity (масштаб амплитуд)
    amps = amps * (0.3 + 0.7*vel)

    return amps.astype(np.float32), noise, env

class HarmonicNet(nn.Module):
    def __init__(self, n_harm=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, n_harm+1),
        )
    def forward(self, f0_hz, vel):
        x = torch.stack([torch.log(f0_hz), vel], dim=-1)
        out = self.net(x)
        harm = torch.nn.functional.softplus(out[...,:n_harm])
        noise = torch.nn.functional.softplus(out[...,-1:])
        scale = harm.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
        return harm/scale, noise/scale[..., :1]

def train_model(instr="bass", epochs=200, seed=0):
    torch.manual_seed(seed)
    model = HarmonicNet(n_harm).to(device)
    opt = optim.Adam(model.parameters(), lr=3e-3)

    if instr == "bass": midi_min, midi_max = 28, 40
    else:               midi_min, midi_max = 60, 72

    for ep in range(epochs):
        B = 128
        midi = torch.randint(midi_min, midi_max+1, (B,)).float()
        f0 = torch.tensor([midi_to_hz(m.item()) for m in midi], dtype=torch.float32)
        vel = torch.rand_like(f0)*0.6 + 0.4

        amps_t = []; noise_t = []
        for f, v in zip(f0.numpy(), vel.numpy()):
            a, n, _ = teacher_params(instr, f, v)
            amps_t.append(a); noise_t.append([n])
        amps_t = torch.tensor(np.stack(amps_t), dtype=torch.float32)
        noise_t = torch.tensor(np.stack(noise_t), dtype=torch.float32)

        harm, noise = model(f0, vel)
        loss = (harm-amps_t).abs().mean() + 0.2*(noise-noise_t).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def render_note(model, f0_hz, vel, env_params, path):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    harm, noise = model(torch.tensor([f0_hz]), torch.tensor([vel]))
    harm = harm.detach().numpy()[0]; noise = float(noise.detach().numpy()[0])
    x = np.zeros_like(t)
    for k in range(1, n_harm+1):
        x += harm[k-1] * np.sin(2*np.pi*k*f0_hz*t)
    n = np.random.randn(len(t))
    n = np.convolve(n, np.ones(9)/9, mode='same')
    x = x + noise*0.25*n
    a,d,s,r = env_params
    x *= adsr(t, a=a, d=d, s=s, r=r)
    x = x/ (np.max(np.abs(x))+1e-8) * 0.95
    wavwrite(str(path), sr, (x*32767).astype(np.int16))

def export_scale(instr="bass", out_dir="out_wavs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model = train_model(instr, epochs=200, seed=1 if instr=="bass" else 2)
    if instr == "bass":
        notes = range(28, 41)  # E1..E2
    else:
        notes = range(60, 73)  # C4..C5
    for m in notes:
        f0 = midi_to_hz(m)
        _, _, envp = teacher_params(instr, f0, 0.8)
        render_note(model, f0, 0.8, envp, Path(out_dir)/f"{instr}_midi{m}.wav")

if __name__ == "__main__":
    export_scale("bass",   "wavs_bass")
    export_scale("trumpet","wavs_trumpet")
