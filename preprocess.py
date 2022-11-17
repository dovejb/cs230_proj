import librosa
import os
import stempeg
import pickle
import numpy as np
from constants import *

def split_wav(wav, seglen):
    length = wav.shape[0]
    N = length // seglen + 1
    wav = np.pad(wav, [(0, N*seglen-length),(0,0)])
    return np.reshape(wav, (N,seglen,1))
            
            
# Split music data into 16s segments
class Preprocess():
    def __init__(self,
                 input_dir=".",
                 output_dir=".",
                 ext=".mp4",
                 segment_duration=32,
                ):
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.ext=ext
        self.segment_duration=segment_duration
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run(self):
        self.files=[]
        filenames = sorted(os.listdir(self.input_dir))
        fno = 0
        for f in filenames:
            if not f.endswith(self.ext):
                continue
            fno += 1
            print(f"Handling {fno} / {len(filenames)}, {f}")
            self.preprocess(f, fno)

    # preprocess 1 file
    def preprocess(self, f, fno):
            rpath = os.path.join(self.input_dir,f) 
            mix, _ = stempeg.read_stems(rpath, stem_id=[0])
            voc, _ = stempeg.read_stems(rpath, stem_id=[4])
            length = len(mix)
            seglen = 4 * 44100  # 4s per segment
            N = length // seglen + 1
            for i in range(N):
                st = i * seglen
                ed = (i+1) * seglen
                if ed > length:
                    ed = length
                stempeg.write_audio(os.path.join(self.output_dir, f"mix_{fno}_{i}_orig.wav"), mix[st:ed], output_sample_rate=SR)
                stempeg.write_audio(os.path.join(self.output_dir, f"voc_{fno}_{i}_orig.wav"), voc[st:ed], output_sample_rate=SR)

    def dump_wav(self, path):
        dir = os.path.dirname(path)
        fname = os.path.basename(path)
        oname = os.path.join(dir, fname[:-4] + ".wav")
        with open(path, 'rb') as f:
            s = pickle.load(f)
        print(s)
        #stempeg.write_audio(oname, s)
                
if __name__ == '__main__':
    Preprocess(input_dir="e:/stanford/cs230/datasets/musdb18/test",
                output_dir="i:/dl/test",
                ).run()
    Preprocess(input_dir="e:/stanford/cs230/datasets/musdb18/train",
                output_dir="i:/dl/train",
                ).run()
    #p.dump_wav("i:/dl/train/mix_1_5_orig.pkl")
    