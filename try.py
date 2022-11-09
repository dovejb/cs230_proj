import tensorflow as tf
import soundfile as sf

s, rate = sf.read("i:/dl/train/voc_46_11_orig.wav")
print(s.shape, rate)
print(s)

s, rate = sf.read("i:/dl/train/voc_46_11_orig.wav", always_2d=True)
print(s.shape, rate)
print(s)
