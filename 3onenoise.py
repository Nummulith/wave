import numpy as np
import librosa
from scipy.io.wavfile import write as wavwrite
from scipy import signal
from pathlib import Path

sr = 22050
dur = 1.0
n_harm = 32   # можно увеличить, но мы будем ограничивать по Nyquist

def midi_to_hz(m):
    return 440.0 * (2 ** ((m - 69.0)/12.0))

# --- Utilities -------------------------------------------------
def parabolic_interpolation(mag, idx):
    # Quadratic interpolation of log-magnitude around idx (idx integer)
    # returns interpolated amplitude (linear) and refined bin position
    if idx <= 0 or idx >= len(mag)-1:
        return mag[idx], float(idx)
    alpha, beta, gamma = mag[idx-1], mag[idx], mag[idx+1]
    p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
    refined = idx + p
    # interpolate magnitude in log domain to reduce bias
    interp_mag = beta - 0.25*(alpha - gamma)*p
    return max(0.0, interp_mag), refined

def lowpass_filter(x, cutoff=9000.0, sr=22050, order=4):
    nyq = 0.5*sr
    normal_cutoff = min(cutoff/nyq, 0.999)
    b,a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, x)

# --- Extract template from one wav (with interpolation & smoothing) ---
def extract_template(wav_path, f0_hz, n_harm=16, sr_global=sr):
    y, sr_loaded = librosa.load(wav_path, sr=sr_global)
    # optional pre-filter the input to remove extreme HF before analysis
    y = lowpass_filter(y, cutoff=min(12000, sr_loaded//2 - 200), sr=sr_loaded, order=6)

    N = len(y)
    window = np.hanning(N)
    Y = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(N, 1/sr_loaded)
    mag = np.abs(Y)

    amps = np.zeros(n_harm, dtype=np.float32)
    for k in range(1, n_harm+1):
        f_h = f0_hz * k
        if f_h >= sr_loaded/2:
            amps[k-1] = 0.0
            continue
        # find nearest bin
        idx = np.argmin(np.abs(freqs - f_h))
        # take a local neighborhood mag to reduce bin noise
        # use parabolic interpolation for amplitude/position refinement
        interp_mag, _ = parabolic_interpolation(mag, idx)
        # also average a few bins around idx to be robust
        neigh = mag[max(0, idx-2): min(len(mag), idx+3)]
        amps[k-1] = max(interp_mag, np.mean(neigh))

    # smooth amplitudes across harmonics (reduces jagged high-frequency content)
    kernel = np.ones(3)/3.0
    amps = np.convolve(amps, kernel, mode='same')
    if amps.max() > 0:
        amps /= amps.max()

    # noise estimate: residual energy excluding the resolved harmonic bins
    # construct harmonic mask (simple narrow-band exclusion)
    mask = np.zeros_like(mag, dtype=bool)
    for k in range(1, n_harm+1):
        f_h = f0_hz * k
        if f_h >= sr_loaded/2: break
        idx = np.argmin(np.abs(freqs - f_h))
        mask[max(0, idx-2):min(len(mag), idx+3)] = True
    harmonic_energy = np.sum(mag[mask]**2)
    total_energy = np.sum(mag**2) + 1e-12
    noise = float(max(0.0, (total_energy - harmonic_energy) / total_energy))

    # estimate ADSR from RMS
    env = estimate_adsr(y, sr_loaded)

    return amps.astype(np.float32), noise, env

# --- Synthesis with harmonic limit, smoothing and output LPF ---
def synthesize_note(f0_hz, amps, noise, env, out_path, sr_global=sr, dur_local=dur):
    t = np.linspace(0, dur_local, int(sr_global*dur_local), endpoint=False)
    x = np.zeros_like(t)

    # only include harmonics below Nyquist to avoid aliasing
    max_k = min(len(amps), int((sr_global/2) // f0_hz))
    if max_k <= 0:
        # silent if fundamental is already above Nyquist (very unlikely here)
        wavwrite(str(out_path), sr_global, (x*32767).astype(np.int16))
        return

    # apply a gentle high-harmonic roll-off to avoid harsh HF
    roll = np.linspace(1.0, 0.0, len(amps))
    for k in range(1, max_k+1):
        amp = amps[k-1] * roll[k-1]
        x += amp * np.sin(2*np.pi*k*f0_hz*t)

    # add noise but lowpass it and scale down
    n = np.random.randn(len(t))
    n = np.convolve(n, np.ones(9)/9, mode='same')  # smooth noise
    # lowpass noise to prevent HF
    n = lowpass_filter(n, cutoff=min(8000, sr_global//2 - 200), sr=sr_global, order=4)
    x += noise * 0.15 * n   # << уменьшил множитель шума (0.15)

    # ADSR envelope (safely build to match length)
    a,d,s,r = env
    a_samps = int(max(1, round(a*sr_global)))
    d_samps = int(max(1, round(d*sr_global)))
    r_samps = int(max(1, round(r*sr_global)))
    sustain_samps = max(1, int(len(t) - (a_samps + d_samps + r_samps)))
    env_curve = np.concatenate([
        np.linspace(0,1,a_samps,endpoint=False),
        np.linspace(1,s,d_samps,endpoint=False),
        np.full(sustain_samps, s),
        np.linspace(s,0,r_samps,endpoint=True)
    ])
    env_curve = env_curve[:len(t)]
    x *= env_curve

    # normalize (but leave headroom)
    x /= (np.max(np.abs(x))+1e-8)
    x *= 0.95

    # final low-pass filter to remove remaining HF artifacts / aliasing
    x = lowpass_filter(x, cutoff=min(9000, sr_global//2 - 200), sr=sr_global, order=6)

    # write
    wavwrite(str(out_path), sr_global, (x*32767).astype(np.int16))

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
