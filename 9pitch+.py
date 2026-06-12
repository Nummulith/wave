import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

def pitch_shift_resample(signal, semitones):
    """
    Сдвиг высоты звука через ресемплинг.
    semitones: число полутонов (+ вверх, - вниз)
    """
    ratio = 2 ** (semitones / 12.0)  # множитель частоты
    N = len(signal)
    new_length = int(N / ratio)  # пересчёт длины
    shifted = resample(signal, new_length)
    return shifted

# Нормализация для WAV
def normalize(sig):
    sig = sig / np.max(np.abs(sig))
    return (sig * 32767).astype(np.int16)

# -----------------
# Пример использования
# -----------------
file_name = "dist_bass" + "_" + "ch0"
fft_data = np.loadtxt(f"fft/{file_name}.csv", delimiter=',')
complex_fft = fft_data[:,0] + 1j * fft_data[:,1]
signal = np.fft.ifft(complex_fft).real
sample_rate = 44100

folder_path = f"set/{file_name}"
os.makedirs(folder_path, exist_ok=True)

for semitones in range(-12, 13):
    shifted = pitch_shift_resample(signal, semitones)
    shifted_wav = normalize(shifted)

    out = f"set/{file_name}/{file_name}-{semitones + 12}.wav"
    wavfile.write(out, sample_rate, shifted_wav)
    print(f"Файл сохранён: {out}")
