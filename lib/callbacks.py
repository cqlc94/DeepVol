import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class EarlyStopping():
    def __init__(self, out_path='best.pt', patience=100, delta=1e-8, **kwargs):
        self.delta = delta 
        self.patience = patience
        self.out_path = out_path

        self.counter = 0
        self.best_epoch = 0
        self.best_loss = np.Inf
        self.stop_flag = False

    def __call__(self, epoch, loss, model):
        if loss < self.best_loss + self.delta:
            self.counter = 0
            self.best_epoch = epoch
            self.best_loss = loss
            torch.save(model.state_dict(), self.out_path)
        else:
            self.counter += 1 
            if self.counter >= self.patience: 
                self.stop_flag = True
        return self.stop_flag
    
class Log():
    def __init__(self, out_path='log.txt'):
        self.out_path = open(out_path, 'a')
        self.log = defaultdict(list)

    def __call__(self, key, value):
        self.log[key].append(value)
    
    def get(self, key, idx=-1):
        return self.log[key][idx]
    
    def print(self, epoch=-1):
        string = f'epoch: {self.get("epoch", epoch)}, '
        for key in self.log.keys():
            if key != 'epoch':
                string += f'{key}: {self.get(key, epoch):.4f}, '
        print(string, file=self.out_path, flush=True)

    