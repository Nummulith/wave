import numpy as np
import matplotlib.pyplot as plt

# Загружаем CSV
csv = 'fft/dist_bass_ch0.csv'
fft_data = np.loadtxt(csv, delimiter=',')
real = fft_data[:,0]
imag = fft_data[:,1]

amplitude = np.sqrt(real**2 + imag**2)
phase = np.arctan2(imag, real)

# Создаем одно окно с двумя графиками
fig, axs = plt.subplots(2, 1, figsize=(12, 8))  # 2 строки, 1 столбец

# График амплитуды
axs[0].plot(amplitude)
axs[0].set_title("Амплитуда спектра сигнала")
axs[0].set_xlabel("Индекс частоты k")
axs[0].set_ylabel("Амплитуда")

# График фазы
axs[1].plot(phase)
axs[1].set_title("Фаза спектра сигнала")
axs[1].set_xlabel("Индекс частоты k")
axs[1].set_ylabel("Фаза (рад)")

plt.tight_layout()  # чтобы графики не накладывались
plt.show()
