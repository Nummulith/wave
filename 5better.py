"""
neural_synth_one_note_to_scale.py

Берёт одну WAV ноту (одну записанную ноту инструмента), анализирует её
и экспортирует ноты для диапазона MIDI 12..47 (0..2 октавы) двумя способами:
  1) FFT-based harmonic template (интерполированные + сглаженные + anti-alias)
  2) LPC-based spectral envelope (оценка огибающей спектра через LPC)

Выходы:
  ./out_fft/<instr>_midi{m}.wav
  ./out_lpc/<instr>_midi{m}.wav

Требования: numpy, librosa, scipy

Пример:
  python neural_synth_one_note_to_scale.py bass.wav --midi-ref 28 --name bass

Если --midi-ref не передан, частота f0 будет найдена автоматически (librosa.yin).
"""

import argparse
from pathlib import Path
import numpy as np
import librosa
from scipy.io.wavfile import write as wavwrite
from scipy import signal
import scipy.linalg as linalg

# --------------------------- CONFIG ---------------------------------
SAMPLE_RATE = 22050
DURATION = 1.0        # длина генерируемой ноты в сек
N_HARM = 64           # увеличено для FFT анализа
MIDI_MIN = 12         # C0-ish
MIDI_MAX = 47         # ~2nd octave

# --------------------------- UTILITIES ------------------------------

def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69.0)/12.0))


def hz_to_midi(hz):
    return 69 + 12*np.log2(hz/440.0)


# Parabolic interpolation for spectral peak refinement
def parabolic_interpolation(mag, idx):
    if idx <= 0 or idx >= len(mag)-1:
        return mag[idx], float(idx)
    alpha, beta, gamma = mag[idx-1], mag[idx], mag[idx+1]
    denom = (alpha - 2*beta + gamma)
    if np.abs(denom) < 1e-12:
        return beta, float(idx)
    p = 0.5 * (alpha - gamma) / denom
    refined = idx + p
    interp_mag = beta - 0.25*(alpha - gamma)*p
    return max(0.0, interp_mag), refined


def butter_lowpass_filter(x, cutoff, sr=SAMPLE_RATE, order=4):
    nyq = 0.5*sr
    normal_cutoff = min(cutoff/nyq, 0.999)
    b,a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, x)


# --------------------------- ADSR estimation ------------------------

def estimate_adsr(y, sr=SAMPLE_RATE):
    frame_len = int(0.01*sr)
    hop = max(1, frame_len // 2)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    t = np.linspace(0, len(y)/sr, len(rms))
    if rms.max() <= 0:
        return 0.01, 0.08, 0.7, 0.15
    rms = rms / (rms.max() + 1e-12)
    peak_idx = np.argmax(rms)
    a_time = max(0.005, (peak_idx * hop) / sr)
    target = 0.7
    after_peak = rms[peak_idx:]
    if after_peak.size == 0:
        d_time = 0.08
    else:
        below = np.where(after_peak < target)[0]
        d_time = ((below[0] * hop) / sr) if below.size>0 else 0.08
    sustain_start = peak_idx + int((d_time*sr)//hop)
    sustain_region = rms[sustain_start: max(sustain_start+3, len(rms)//2)]
    sustain_level = float(np.median(sustain_region)) if sustain_region.size>0 else 0.7
    sustain_level = float(np.clip(sustain_level, 0.05, 0.98))
    rev = rms[::-1]
    threshold = sustain_level * 0.15
    below_end = np.where(rev < threshold)[0]
    r_time = ((below_end[0] * hop) / sr) if below_end.size>0 else 0.2
    return float(a_time), float(d_time), float(sustain_level), float(r_time)


# --------------------------- FFT-BASED TEMPLATE ---------------------

def extract_template_fft(wav_path, f0_hz, n_harm=N_HARM, sr=SAMPLE_RATE):
    y, sr_loaded = librosa.load(wav_path, sr=sr)
    cutoff_pref = min(12000.0, sr_loaded/2 - 200)
    if cutoff_pref > 1000:
        y = butter_lowpass_filter(y, cutoff_pref, sr=sr_loaded, order=6)
    N = len(y)
    center = N // 2
    wlen = int(0.5 * sr_loaded)
    start = max(0, center - wlen//2)
    end = min(N, start + wlen)
    frame = y[start:end]
    window = np.hanning(len(frame))
    fft_size = 2**int(np.ceil(np.log2(len(frame)))) * 4
    Y = np.fft.rfft(frame * window, n=fft_size)
    freqs = np.fft.rfftfreq(fft_size, 1/sr_loaded)
    mag = np.abs(Y)
    amps = np.zeros(n_harm, dtype=np.float32)
    for k in range(1, n_harm+1):
        f_h = f0_hz * k
        if f_h >= sr_loaded/2:
            amps[k-1] = 0.0
            continue
        idx = np.argmin(np.abs(freqs - f_h))
        interp_mag, _ = parabolic_interpolation(mag, idx)
        neigh = mag[max(0, idx-3): min(len(mag), idx+4)]
        amps[k-1] = max(interp_mag, np.mean(neigh))
    kernel = np.ones(5)/5.0
    amps = np.convolve(amps, kernel, mode='same')
    if amps.max() > 0:
        amps /= (amps.max() + 1e-12)
    mask = np.zeros_like(mag, dtype=bool)
    for k in range(1, n_harm+1):
        f_h = f0_hz * k
        if f_h >= sr_loaded/2: break
        idx = np.argmin(np.abs(freqs - f_h))
        mask[max(0, idx-3):min(len(mag), idx+4)] = True
    harmonic_energy = np.sum(mag[mask]**2)
    total_energy = np.sum(mag**2) + 1e-12
    noise = float(max(0.0, (total_energy - harmonic_energy) / total_energy))
    env = estimate_adsr(y, sr_loaded)
    return amps.astype(np.float32), noise, env


# --------------------------- LPC-BASED TEMPLATE ---------------------

def lpc_coefficients(x, order):
    r = np.correlate(x, x, mode='full')
    mid = len(r)//2
    r = r[mid: mid+order+1]
    R = linalg.toeplitz(r[:-1])
    rhs = -r[1:]
    try:
        a = linalg.solve(R, rhs)
        coeffs = np.concatenate(([1.0], a))
    except Exception:
        coeffs = np.zeros(order+1)
        coeffs[0] = 1.0
    return coeffs


def extract_template_lpc(wav_path, f0_hz, n_harm=N_HARM, sr=SAMPLE_RATE, lpc_order=32):
    y, sr_loaded = librosa.load(wav_path, sr=sr)
    N = len(y)
    center = N//2
    wlen = min(N, int(0.2*sr_loaded))
    start = max(0, center - wlen//2)
    end = min(N, start + wlen)
    frame = y[start:end]
    frame = signal.lfilter([1.0, -0.97], [1.0], frame)
    coeffs = lpc_coefficients(frame, lpc_order)
    w, h = signal.freqz(1.0, coeffs, worN=4096, fs=sr_loaded)
    env_mag = np.abs(h)
    env_mag = env_mag / (env_mag.max() + 1e-12)
    amps = np.zeros(n_harm, dtype=np.float32)
    for k in range(1, n_harm+1):
        f_h = f0_hz * k
        if f_h >= sr_loaded/2:
            amps[k-1] = 0.0
            continue
        idx = np.argmin(np.abs(w - f_h))
        neigh = env_mag[max(0, idx-2): min(len(env_mag), idx+3)]
        amps[k-1] = np.mean(neigh)
    amps = np.convolve(amps, np.ones(3)/3.0, mode='same')
    if amps.max() > 0:
        amps /= (amps.max() + 1e-12)
    noise = 0.03
    env = estimate_adsr(y, sr_loaded)
    return amps.astype(np.float32), float(noise), env

# --------------------------- SYNTHESIS -----------------------------

def synthesize_note(amps, noise, env, f0_hz, out_path, sr=SAMPLE_RATE, dur=DURATION):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    x = np.zeros_like(t)

    # limit harmonics under Nyquist
    max_k = min(len(amps), int((sr/2) // f0_hz))
    if max_k <= 0:
        wavwrite(str(out_path), sr, (x*32767).astype(np.int16))
        return

    # gentle roll-off for high harmonics
    roll = np.linspace(1.0, 0.0, len(amps))

    for k in range(1, max_k+1):
        amp = float(amps[k-1]) * float(roll[k-1])
        x += amp * np.sin(2*np.pi*k*f0_hz*t)

    # add smoothed, lowpassed noise
    n = np.random.randn(len(t))
    n = np.convolve(n, np.ones(9)/9, mode='same')
    n = butter_lowpass_filter(n, cutoff=min(8000, sr//2 - 200), sr=sr, order=4)
    x += noise * 0.12 * n

    # ADSR
    a,d,s,r = env
    a_samps = max(1, int(round(a*sr)))
    d_samps = max(1, int(round(d*sr)))
    r_samps = max(1, int(round(r*sr)))
    sustain_samps = max(1, int(len(t) - (a_samps + d_samps + r_samps)))
    env_curve = np.concatenate([
        np.linspace(0,1,a_samps,endpoint=False),
        np.linspace(1,s,d_samps,endpoint=False),
        np.full(sustain_samps, s),
        np.linspace(s,0,r_samps,endpoint=True)
    ])
    env_curve = env_curve[:len(t)]
    x *= env_curve

    # normalize and final LPF
    x = x / (np.max(np.abs(x)) + 1e-12)
    x *= 0.95
    x = butter_lowpass_filter(x, cutoff=min(9000, sr//2 - 200), sr=sr, order=6)

    wavwrite(str(out_path), sr, (x*32767).astype(np.int16))


# --------------------------- MAIN EXPORT ---------------------------

def export_from_one(wav_path, midi_ref=None, name="instr"):
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"File not found: {wav_path}")

    # load once to detect f0 if needed
    y_full, sr_loaded = librosa.load(str(wav_path), sr=SAMPLE_RATE)
    if midi_ref is None:
        # try to detect f0 using librosa.yin over the file (robust to single note)
        f0_series = librosa.yin(y_full, fmin=50, fmax=1000, sr=sr_loaded)
        # pick median non-NaN
        f0_vals = f0_series[~np.isnan(f0_series)]
        if f0_vals.size == 0:
            raise RuntimeError("Could not detect f0 automatically; provide --midi-ref")
        f0_ref = float(np.median(f0_vals))
        midi_ref = int(round(hz_to_midi(f0_ref)))
    else:
        f0_ref = midi_to_hz(midi_ref)

    print(f"Reference f0: {f0_ref:.2f} Hz (MIDI {midi_ref})")

    # FFT template
    amps_fft, noise_fft, env_fft = extract_template_fft(str(wav_path), f0_ref, n_harm=N_HARM, sr=SAMPLE_RATE)
    # LPC template
    amps_lpc, noise_lpc, env_lpc = extract_template_lpc(str(wav_path), f0_ref, n_harm=N_HARM, sr=SAMPLE_RATE, lpc_order=32)

    out_fft_dir = Path("out_fft")/name
    out_lpc_dir = Path("out_lpc")/name
    out_fft_dir.mkdir(parents=True, exist_ok=True)
    out_lpc_dir.mkdir(parents=True, exist_ok=True)

    for m in range(MIDI_MIN, MIDI_MAX+1):
        f0 = midi_to_hz(m)
        out_fft = out_fft_dir/f"{name}_midi{m}.wav"
        out_lpc = out_lpc_dir/f"{name}_midi{m}.wav"
        synthesize_note(amps_fft, noise_fft, env_fft, f0, out_fft, sr=SAMPLE_RATE, dur=DURATION)
        synthesize_note(amps_lpc, noise_lpc, env_lpc, f0, out_lpc, sr=SAMPLE_RATE, dur=DURATION)

    print(f"Export finished. FFT outputs -> {out_fft_dir}; LPC outputs -> {out_lpc_dir}")


# --------------------------- CLI ----------------------------------

if __name__ == '__main__':
    # p = argparse.ArgumentParser()
    # p.add_argument('wav', help='input WAV file with single note')
    # p.add_argument('--midi-ref', type=int, default=None, help='MIDI number of the input note (optional)')
    # p.add_argument('--name', default='instr', help='output subfolder name')
    # args = p.parse_args()
    # export_from_one(args.wav, midi_ref=args.midi_ref, name=args.name)

    export_from_one("bass.wav", 12, "bass")
