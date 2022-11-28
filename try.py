import torch.nn.functional as F
import torch

class G:
    x = 1

class A:
    def add(self):
        G.x += 1
    def print(self):
        print(G.x)

if __name__ == '__main__':
    a = A()
    b = A()
    a.print()
    b.print()
    a.add()
    b.print()