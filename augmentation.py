import librosa
import numpy as np


class Augmentation():

    def __init__(self,
                 input_dir=".",
                 output_dir=".",
                 seed=9182781):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.rand = np.random.RandomState(seed=seed)
        self._init_methods_dict()
    
    def generate_plan(self):
        n = self.rand.choice(np.arange(1,4), p=[0.5,0.3,0.2])
        plan = []
        used = {}
        while len(plan) < n:
            m = self.rand.choice(self.method_keys)
            if m in used and used[m]:
                continue
            used[m] = True
            o = self.rand.choice(self.methods[m]['keys'])
            plan.append((m,o))
        return plan

    def generate_plans(self, n):
        plans = []
        for _ in range(n):
            plans.append(self.generate_plan())
        return plans

    def _init_methods_dict(self):
        self.methods = {
            "PITCH": {
                "UP": self.pitch_up,
                "DOWN": self.pitch_down,
            },
            "SPEED": {
                "UP": self.speed_up,
                "DOWN": self.speed_down,
            },
            "GAIN": {
                "UP": self.gain_up,
                "DOWN": self.gain_down,
            },
            "REMIX": {
                "REMIX": self.remix,
            },
            "NOISE": {
                "NOISE": self.noise,
            },
            "FILTER": {
                "HIGHPASS": self.filter_high,
                "LOWPASS": self.filter_low,
                "BANDPASS": self.filter_band,
            },
            "IMPULSE": {
                "IMPLUSE": self.impluse,
            },
            "TIME": {
                "FORWARD": self.time_forward,
                "BACKWARD": self.time_backward,
            },
        }
        self.method_keys = [k for k in self.methods]
        for m in self.methods.values():
            m["keys"] = [o for o in m]

    def pitch_up(self):
        pass
    def pitch_down(self):
        pass
    def speed_up(self):
        pass
    def speed_down(self):
        pass
    def gain_up(self):
        pass
    def gain_down(self):
        pass
    def remix(self):
        pass
    def noise(self):
        pass
    def filter_high(self):
        pass
    def filter_low(self):
        pass
    def filter_band(self):
        pass
    def impluse(self):
        pass
    def time_forward(self):
        pass
    def time_backward(self):
        pass


if __name__ == '__main__':
    a = Augmentation()
    for p in a.generate_plans(10):
        print(p)