import numpy as np
from scipy.io import wavfile
from wave_lib import *

file_name = 'dist_bass'
sou_name = f"source/{file_name}.wav"

source = Wave()
source.load_wav(sou_name)

csv_format = False

fft_name = f"fft/{file_name}_ch[ch].{'npy' if not csv_format else 'csv'}"
saver = source.save_fft if not csv_format else source.save_csv
saver(fft_name)

dest = Wave()
dest.copy_properties_from(source)

loader = dest.load_fft if not csv_format else dest.load_csv
loader(fft_name)

source.diff_check(dest)

rec_name = f"recovered/{file_name}.wav"
dest.save_wav(rec_name)
