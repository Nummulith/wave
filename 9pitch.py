import os
import numpy as np
from scipy.io import wavfile

def pitch_shift_fft(complex_fft, semitones, sample_rate):
    """
    Сдвиг высоты звука на заданное количество полутонов через масштабирование спектра.
    complex_fft: спектр (комплексный массив после FFT)
    semitones: целое/вещественное число (например, 1 = вверх на полтона, -2 = вниз на тон)
    sample_rate: частота дискретизации
    """
    N = len(complex_fft)
    freqs = np.fft.fftfreq(N, 1.0/sample_rate)

    ratio = 2 ** (semitones / 12.0)  # коэффициент частоты
    new_fft = np.zeros_like(complex_fft)

    # переносим спектр
    for i, f in enumerate(freqs):
        new_f = f * ratio
        j = int(np.round(new_f / (sample_rate / N)))  # в какой бин попадет новая частота
        if -N//2 <= j < N//2:
            new_fft[j % N] += complex_fft[i]

    return new_fft

# -----------------
# Пример использования
# -----------------
file_name = "dist_bass" + "_" + "ch0"
fft_data = np.loadtxt(f"fft/{file_name}.csv", delimiter=',')
complex_fft = fft_data[:,0] + 1j * fft_data[:,1]
sample_rate = 44100

folder_path = f"set/{file_name}"
os.makedirs(folder_path, exist_ok=True)

for semitones in range(-12, 13):

    # Сдвиг на полутона
    shifted_fft = pitch_shift_fft(complex_fft, semitones=semitones, sample_rate=sample_rate)
    shifted_data = np.fft.ifft(shifted_fft).real

    # Нормализация
    def normalize(sig):
        sig = sig / np.max(np.abs(sig))
        return (sig * 32767).astype(np.int16)

    shifted_wav = normalize(shifted_data)

    out = f"set/{file_name}/{file_name}-{semitones + 12}.wav"
    wavfile.write(out, sample_rate, shifted_wav)

    print(f"Файл сохранён: {out}")
