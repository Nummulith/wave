import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

class Wave:
    def load_wav(self, file_name):
        self.sample_rate, data = wavfile.read(file_name)
        self.data_float = data.astype(np.float64)
        print(f"file read: {file_name} - sample_rate = {self.sample_rate}")

        # Если моно, делаем (N,1)
        if self.data_float.ndim == 1:
            self.data_float = self.data_float[:, np.newaxis]
        self.num_channels = self.data_float.shape[1]
        self.dtype = data.dtype

    def file_name_ch(self, file_name, ch):
        return file_name.replace("[ch]", str(ch))

    def save_fft(self, file_name):
        for ch in range(self.num_channels):
            fft_data = np.fft.fft(self.data_float[:, ch])
            np.save(
                self.file_name_ch(file_name, ch),
                fft_data
            )

    def save_csv(self, file_name):
        for ch in range(self.num_channels):
            fft_data = np.fft.fft(self.data_float[:, ch])
            np.savetxt(
                self.file_name_ch(file_name, ch),
                np.column_stack((fft_data.real, fft_data.imag)), delimiter=',', fmt='%.16f'
            )

    def copy_properties_from(self, source):
        self.data_float = np.zeros_like(source.data_float)
        self.num_channels = source.num_channels
        self.dtype = source.dtype
        self.sample_rate = source.sample_rate

    def diff_check(self, source):
        check = np.max(np.abs(self.data_float - source.data_float))
        print("Difference check: max = ", check)

    def load_fft(self, file_name):
        for ch in range(self.num_channels):
            fft_complex = np.load(self.file_name_ch(file_name, ch))
            self.data_float[:, ch] = np.fft.ifft(fft_complex).real

    def load_csv(self, file_name):
        for ch in range(self.num_channels):
            fft_loaded = np.loadtxt(self.file_name_ch(file_name, ch), delimiter=',')
            fft_complex = fft_loaded[:, 0] + 1j * fft_loaded[:, 1]
            self.data_float[:, ch] = np.fft.ifft(fft_complex).real

    def save_wav(self, file_name):
        data = self.data_float
        data = (data / np.max(np.abs(data))) * 32767
        data = np.round(data)
        data = data.astype(self.dtype) # Округляем и возвращаем исходный тип данных
        if self.num_channels == 1: # Если был моно, убираем лишнюю ось
            data = data[:, 0]

        wavfile.write(file_name, self.sample_rate, data)

    def resample(self, semitones):
        """
        Сдвиг высоты звука через ресемплинг.
        semitones: число полутонов (+ вверх, - вниз)
        """
        ratio = 2 ** (semitones / 12.0)  # множитель частоты
        new_length = int(len(self.data_float) / ratio)  # пересчёт длины

        resampled = []
        for ch in range(self.data_float.shape[1]):
            resampled.append(resample(self.data_float[:, ch], new_length))
        self.data_float = np.array(resampled).T  # Транспонируем обратно