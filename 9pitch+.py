import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from wave_lib import *

map = {
    0: "C" , 1: "C#", 2: "D" , 3: "D#",  4: "E",  5: "F",
    6: "F#", 7: "G" , 8: "G#", 9: "A" , 10: "B", 11: "H",
}

file_name = "dist_bass"
sou_name = f"source/{file_name}.wav"

folder_path = f"set/{file_name}"
os.makedirs(folder_path, exist_ok=True)

for semitones in range(-12, 13):
    sem  = semitones + 12
    oct  = sem // 12
    note = map[sem % 12]

    # out = f"set/{file_name}/{file_name}-{semitones + 12}.wav"
    out = f"set/{file_name}/{oct}-{note}.wav"

    source = Wave()
    source.load_wav(sou_name)
    source.resample(semitones)
    source.save_wav(out)
    print(f"Файл сохранён: {out}")
