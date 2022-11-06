import librosa
import os
import stempeg

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

        self.load_files()
    
    def load_files(self):
        self.files=[]
        files = os.listdir(self.input_dir)
        for f in files:
            if not f.endswith(self.ext):
                continue
            info = stempeg.Info(os.path.join(self.input_dir,f))
            print(info)
            break
if __name__ == '__main__':
    p = Preprocess(input_dir="e:/stanford/cs230/datasets/musdb18/train")