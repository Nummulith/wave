import numpy as np
from scipy.io import wavfile

# Загружаем FFT из csv
csv = "bass_ch0.csv"
fft_data = np.loadtxt(csv, delimiter=',')

# Восстанавливаем комплексные значения
complex_fft = fft_data[:,0] + 1j * fft_data[:,1]

# -----------------
# Обработка спектра
# -----------------

# Пример 1: увеличить амплитуду
louder_fft = complex_fft * 0.7

# Пример 2: изменить тон (сдвинуть спектр по частоте)
k = 100  # на сколько "биннов" сдвинуть
pitch_fft = np.roll(complex_fft, k)

# Пример 3: изменить тембр (обрезаем высокие частоты)
sample_rate = 44100
N = len(complex_fft)
freqs = np.fft.fftfreq(N, 1.0/sample_rate)

timbre_fft = complex_fft.copy()
timbre_fft[np.abs(freqs) > 5000] = 0  # LPF до 5 кГц

# -----------------
# Восстановление сигналов
# -----------------
louder_data = np.fft.ifft(louder_fft).real
pitch_data = np.fft.ifft(pitch_fft).real
timbre_data = np.fft.ifft(timbre_fft).real

# -----------------
# Нормализация для wav
# -----------------
def normalize(sig):
    sig = sig / np.max(np.abs(sig))  # в диапазон [-1..1]
    return (sig * 32767).astype(np.int16)  # 16-bit PCM

louder_wav = normalize(louder_data)
pitch_wav = normalize(pitch_data)
timbre_wav = normalize(timbre_data)

# -----------------
# Сохраняем WAV
# -----------------
wavfile.write("drumroll_louder.wav", sample_rate, louder_wav)
wavfile.write("drumroll_pitch.wav", sample_rate, pitch_wav)
wavfile.write("drumroll_timbre.wav", sample_rate, timbre_wav)

print("Файлы сохранены: drumroll_louder.wav, drumroll_pitch.wav, drumroll_timbre.wav")
