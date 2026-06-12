import numpy as np
from scipy.io import wavfile

# -----------------
# Настройки
# -----------------
name = "MobyUp" + "_" + "ch0"
fft_data = np.loadtxt(name + ".csv", delimiter=',')

# Восстанавливаем спектр
complex_fft = fft_data[:,0] + 1j * fft_data[:,1]
sample_rate = 44100
N = len(complex_fft)
freqs = np.fft.fftfreq(N, 1.0/sample_rate)

# Нормализация для WAV
def normalize(sig):
    sig = sig / np.max(np.abs(sig))
    return (sig * 32767).astype(np.int16)

# -----------------
# Мягкий Band-Stop фильтр
# -----------------
def soft_band_stop(fft, freqs, f_start, l_start, min_gain, f_end, l_end):
    """
    Плавное подавление частот в диапазоне [f_start, f_end] до коэффициента min_gain.
    Вне диапазона — gain=1
    """
    fft_filtered = fft.copy()
    abs_freqs = np.abs(freqs)
    
    gain = np.ones_like(fft)
    
    # Плавное подавление с плато
    gain = np.ones_like(fft)
    abs_freqs = np.abs(freqs)

    # Левая зона: линейное уменьшение от 1 до min_gain
    mask_left = (abs_freqs >= f_start) & (abs_freqs < f_start_min)
    gain[mask_left] = 1 - (1 - min_gain) * (abs_freqs[mask_left] - f_start) / (f_start_min - f_start)

    # Средняя зона: ровно min_gain
    mask_middle = (abs_freqs >= f_start_min) & (abs_freqs <= f_end_min)
    gain[mask_middle] = min_gain

    # Правая зона: линейное увеличение от min_gain до 1
    mask_right = (abs_freqs > f_end_min) & (abs_freqs <= f_end)
    gain[mask_right] = min_gain + (1 - min_gain) * (abs_freqs[mask_right] - f_end_min) / (f_end - f_end_min)

    # Применяем к спектру
    fft_filtered *= gain

    return fft_filtered

filters = [
    ('LPF', 1500, 2500, 0.01,  N//2,  N//2), # вырезаем всё выше
    ('HPF',    0,    0, 0.01, 12500, 13500), # вырезаем всё ниже
    ('BSF',  500, 3500, 0.01,  6500,  9500), # вырезаем средние
]

for mode, f_start, f_start_min, min_gain, f_end_min, f_end in filters:
    fft_filtered = soft_band_stop(complex_fft, freqs, f_start, f_start_min, min_gain, f_end_min, f_end)
    data_filtered = np.fft.ifft(fft_filtered).real
    
    out = f"out/{name}_{mode}.wav" # ({f_start}-{f_end})
    wavfile.write(out, sample_rate, normalize(data_filtered))
    print(f"Файл сохранён: {out}")
