import numpy as np
import librosa
from scipy.io.wavfile import write as wavwrite
from pathlib import Path

sr = 22050
dur = 1.0
n_harm = 16

def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69.0)/12.0))

def adsr_from_rms(y, sr):
    # RMS огибающая
    frame_len = int(0.02*sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=frame_len)[0]
    t = np.linspace(0, len(y)/sr, len(rms))
    # очень грубая аппроксимация ADSR
    a = 0.01
    d = 0.08
    s = 0.7
    r = 0.2
    return a, d, s, r

def extract_template(wav_path, f0_hz, n_harm=16, sr=22050):
    y, sr_loaded = librosa.load(wav_path, sr=sr)
    N = len(y)
    Y = np.fft.rfft(y * np.hanning(N))
    freqs = np.fft.rfftfreq(N, 1/sr)

    amps = np.zeros(n_harm, dtype=np.float32)
    for k in range(1, n_harm+1):
        f_h = f0_hz * k
        idx = np.argmin(np.abs(freqs - f_h))
        amps[k-1] = np.abs(Y[idx])

    amps /= amps.max() + 1e-8

    harmonic_energy = np.sum(amps**2)
    total_energy = np.sum(np.abs(Y)**2)
    noise = max(0.0, (total_energy - harmonic_energy) / (total_energy+1e-8))

    env = adsr_from_rms(y, sr_loaded)

    return amps, noise, env

def synthesize_note(f0_hz, amps, noise, env, out_path):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    x = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        x += a * np.sin(2*np.pi*k*f0_hz*t)
    # шум
    n = np.random.randn(len(t))
    n = np.convolve(n, np.ones(9)/9, mode='same')
    x += noise * 0.25 * n
    # ADSR
    a,d,s,r = env
    env_curve = np.concatenate([
        np.linspace(0,1,int(a*sr),endpoint=False),
        np.linspace(1,s,int(d*sr),endpoint=False),
        np.full(int(dur*sr) - int((a+d+r)*sr), s),
        np.linspace(s,0,int(r*sr),endpoint=True)
    ])
    env_curve = env_curve[:len(x)]
    x *= env_curve
    # нормализация
    x /= (np.max(np.abs(x))+1e-8) * 0.95
    wavwrite(str(out_path), sr, (x*32767).astype(np.int16))

def export_from_one(wav_path, midi_ref, out_dir="bass"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    f0_ref = midi_to_hz(midi_ref)
    amps, noise, env = extract_template(wav_path, f0_ref, n_harm)

    # 0 октава (midi ~ 12) до 2-й (midi ~ 47)
    for m in range(12, 48):
        f0 = midi_to_hz(m)
        out_path = Path(out_dir)/f"midi{m}.wav"
        synthesize_note(f0, amps, noise, env, out_path)

if __name__ == "__main__":
    # например, bass.wav это нота "E1" = MIDI 28
    export_from_one("bass.wav", midi_ref=28, out_dir="bass")
