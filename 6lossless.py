import numpy as np
from scipy.io import wavfile

def lossless(file_name):
    # 1. Чтение WAV
    sou_name = f"source/{file_name}.wav"
    sample_rate, data = wavfile.read(sou_name)
    data_float = data.astype(np.float64)
    print(f"file read: {sou_name} - sample_rate = {sample_rate}")

    # Если моно, делаем (N,1)
    if data_float.ndim == 1:
        data_float = data_float[:, np.newaxis]
    num_channels = data_float.shape[1]

    # format
    csv_format = False

    # 2. FFT и сохранение каждого канала в отдельный CSV
    for ch in range(num_channels):
        fft_data = np.fft.fft(data_float[:, ch])
        if not csv_format:
            fft_name = f"fft/{file_name}_ch{ch}.npy"
            np.save(fft_name, fft_data)
        else:
            fft_name = f"fft/{file_name}_ch{ch}.csv"
            np.savetxt(fft_name, np.column_stack((fft_data.real, fft_data.imag)), delimiter=',', fmt='%.16f')
        print(f"Channel {ch} FFT saved to: {fft_name}")

    # 3. Восстановление WAV
    recovered = np.zeros_like(data_float)
    for ch in range(num_channels):
        if not csv_format:
            fft_name = f"fft/{file_name}_ch{ch}.npy"
            fft_complex = np.load(fft_name)
        else:
            fft_name = f"fft/{file_name}_ch{ch}.csv"
            fft_loaded = np.loadtxt(fft_name, delimiter=',')
            fft_complex = fft_loaded[:, 0] + 1j * fft_loaded[:, 1]

        recovered[:, ch] = np.fft.ifft(fft_complex).real
        print(f"Channel {ch} FFT recovered from: {fft_name}")

    # Округляем и возвращаем исходный тип данных
    recovered = np.round(recovered).astype(data.dtype)

    # Если был моно, убираем лишнюю ось
    if num_channels == 1:
        recovered = recovered[:, 0]

    # 4. Проверка
    diff_check = np.max(np.abs(data - recovered))
    print("Difference check: max = ", diff_check)

    # 5. Сохранение WAV
    rec_name = f"recovered/{file_name}.wav"
    wavfile.write(rec_name, sample_rate, recovered)
    print(f"Recovered file written: {rec_name}")

file_name = 'dist_bass'
lossless(file_name)
