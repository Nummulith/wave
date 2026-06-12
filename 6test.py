import numpy as np
from scipy.io import wavfile

# 1. Чтение WAV
sample_rate, data = wavfile.read('input.wav')

# 2. FFT
fft_data = np.fft.fft(data.astype(np.float64))

# 3. Сохраняем в бинарный файл (.npy)
np.save('fft_data.npy', fft_data)

# 4. Загружаем обратно
fft_loaded = np.load('fft_data.npy')

# 5. IFFT и приведение к int32
recovered = np.fft.ifft(fft_loaded).real
recovered = np.round(recovered).astype(np.int32)

# 6. Сохраняем WAV
wavfile.write('recovered.wav', sample_rate, recovered)

# 7. Проверка
print("Max difference:", np.max(np.abs(data - recovered)))
