import numpy as np
from scipy.io import wavfile

big_file   = 'C:/Code/p.Music/106/106.wav'
click_file = 'C:/Code/p.Music/106/click.wav'
out_file   = 'C:/Code/p.Music/106/106_noclick.wav'

# читаем файлы
sr_big, big_data = wavfile.read(big_file)
sr_click, click_data = wavfile.read(click_file)

print(f"big: sr={sr_big}, shape={big_data.shape}, dtype={big_data.dtype}")
print(f"click: sr={sr_click}, shape={click_data.shape}, dtype={click_data.dtype}")

if sr_big != sr_click:
    raise ValueError("Разные sample rate — нужно ресемплировать!")

# если файлы моно или стерео — привести их к двумерной форме
if big_data.ndim == 1:
    big_data = big_data[:, np.newaxis]
if click_data.ndim == 1:
    click_data = click_data[:, np.newaxis]

click_len = click_data.shape[0]
N = big_data.shape[0]

# вычитаем метроном на интервалах
interval_samples = 7000  # через сколько сэмплов вычитать
dtype = big_data.dtype
info = np.iinfo(dtype)

# работаем в int64 чтобы не переполнить, потом обратно в int32/int16
big_data64 = big_data.astype(np.int64)
click64    = click_data.astype(np.int64)

for start in range(0, N - click_len, interval_samples):
    big_data64[start:start+click_len, :] -= click64

# ограничить диапазон перед возвратом к исходному типу
big_data64 = np.clip(big_data64, info.min, info.max)
big_data_out = big_data64.astype(dtype)

wavfile.write(out_file, sr_big, big_data_out)
print(f"Сохранили {out_file}")
