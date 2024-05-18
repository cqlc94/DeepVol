import pickle

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

### Fixed context length Dataset
class CDMDataset(Dataset):
    def __init__(self, stage = 'train', n_samples = None, data_dir = None, **kwargs):
        self.setup(stage, n_samples, data_dir)
        
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        return self.input[idx], self.target[idx]

    def setup(self, stage, n_samples, data_dir):
        data_dict = pickle.load(open(data_dir, 'rb'))
        self.rics = data_dict['rics']
        if stage == 'train':
            self.input = data_dict['X_train'][:n_samples]
            self.target = data_dict['y_train'][:n_samples]
        elif stage == 'val':
            self.input = data_dict['X_val'][:n_samples]
            self.target = data_dict['y_val'][:n_samples]
        elif stage == 'test':
            self.input = data_dict['X_test'][:n_samples]
            self.target = data_dict['y_test'][:n_samples]
        elif stage == 'all':
            self.input = data_dict['X'][:n_samples]
            self.target = data_dict['y'][:n_samples]
        else:
            raise ValueError('Invalid stage')

class SimulationDataset(CDMDataset):
    def __init__(self, stage = 'train', n_samples = None, data_dir = None, **kwargs):
        super().__init__(stage, n_samples, data_dir)

    def __getitem__(self, idx):
        return self.input[idx], self.target[idx], self.target2[idx]
        
    def setup(self, stage, n_samples, data_dir):
        data_dict = pickle.load(open(data_dir, 'rb'))
        self.rics = data_dict['rics']
        if stage == 'train':
            self.input = data_dict['X_train'][:n_samples]
            self.target = data_dict['y_train'][:n_samples]
            self.target2 = data_dict['h_train'][:n_samples]
        elif stage == 'val':
            self.input = data_dict['X_val'][:n_samples]
            self.target = data_dict['y_val'][:n_samples]
            self.target2 = data_dict['h_val'][:n_samples]
        elif stage == 'test':
            self.input = data_dict['X_test'][:n_samples]
            self.target = data_dict['y_test'][:n_samples]
            self.target2 = data_dict['h_test'][:n_samples]
        elif stage == 'all':
            self.input = data_dict['X'][:n_samples]
            self.target = data_dict['y'][:n_samples]
            self.target2 = data_dict['h'][:n_samples]
        else:
            raise ValueError('Invalid stage')
        
def collate_fn_padding(batch): # for train/val/testing set with all context
    xb, yb = zip(*batch)
    mkb = [torch.ones_like(y) for y in yb]
    xb = pad_sequence(xb, batch_first=True)
    yb = pad_sequence(yb, batch_first=True)
    mkb = pad_sequence(mkb, batch_first=True).bool() # False = padding
    return xb, yb, mkb

    





