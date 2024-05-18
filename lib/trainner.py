# Stanaard 
import os
import sys
import time
import json
import pickle
import numpy as np
# PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
# Custom 
from . import data_utils, models, callbacks, metrics, schedulers

"""
Base Trainner Class
"""
class BaseTrainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.device)
        self.configure_seeds()
        self.configure_models()
        self.configure_criterions()
        self.configure_dataloaders()
        self.configure_optimizers()
        self.configure_schedulers()
        self.configure_callbacks()

    def configure_seeds(self):
        torch.manual_seed(self.args.seed)
        if self.args.device == 'cuda': torch.cuda.manual_seed(self.args.seed)
    
    def configure_optimizers(self):
        if self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f'Invalid Optimizer: {self.args.optimizer}')

    def configure_schedulers(self):
        if self.args.scheduler == 'constant':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                               lambda x: 1)
            
        elif self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                            step_size=10, 
                                                            gamma=self.args.gamma)

        elif self.args.scheduler == 'exponential':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                                    gamma=self.args.gamma)

        elif self.args.scheduler == 'cosine':
            self.scheduler = schedulers.CustomCosineAnnealingLR(self.optimizer, 
                                                                T_max=self.args.T_max, 
                                                                eta_min=self.args.lr_min)    

        elif self.args.scheduler == 'cosinewarmrestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                                                                  T_0=10, 
                                                                                  T_mult=2, 
                                                                                  eta_min=self.args.lr_min)
        else:
            raise ValueError(f'Invalid Scheduler: {self.args.scheduler}')

    def configure_callbacks(self):
        # Create root directory
        os.makedirs(self.args.out_dir, exist_ok=True) 

        # Save args
        self.args.n_params  = sum(p.numel() for p in self.model.parameters() if p.requires_grad) 
        json.dump(self.args.__dict__, open(f'{self.args.out_dir}/args.json', 'w'), indent=4) # args file

        # Callbacks
        self.log = callbacks.Log(out_path=f'{self.args.out_dir}/log.txt')
        self.early_stopping = callbacks.EarlyStopping(out_path=f'{self.args.out_dir}/best.pt',
                                                      patience=self.args.patience, 
                                                      delta=self.args.delta)

    def configure_criterions(self):
        self.criterion = metrics.nll_loss

    def configure_models(self):
        if self.args.model == 'lstm':
            self.model = models.LSTM(d_in=self.args.d_in,
                                    d_hidden=self.args.d_hidden,
                                    n_layers=self.args.n_layers,
                                    dropout=self.args.dropout)
            
        elif self.args.model == 'rnn':
            self.model = models.RNN(d_in=self.args.d_in,
                                    d_hidden=self.args.d_hidden,
                                    n_layers=self.args.n_layers,
                                    dropout=self.args.dropout)
        
        elif self.args.model == 'gru':
            self.model = models.GRU(d_in=self.args.d_in,
                                    d_hidden=self.args.d_hidden,
                                    n_layers=self.args.n_layers,
                                    dropout=self.args.dropout)
            
        elif self.args.model == 'transformer':
            self.model = models.Transformer(d_in=self.args.d_in,
                                            d_hidden=self.args.d_hidden,
                                            n_layers=self.args.n_layers,
                                            dropout=self.args.dropout,
                                            nhead=self.args.nhead,
                                            d_ff=self.args.d_ff)
            
        elif self.args.model == 'garch':
            self.model = models.GARCH()
            
        else:
            raise ValueError('Invalid Model')
        
        self.model = self.model.to(self.device)

    def configure_dataloaders(self):
        pass
    

"""
Empirical Trainner: the main difference with Simulation is Empirical trainner need to handle data with different length
"""
class EmpiricalTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def configure_dataloaders(self):
        # Dataset
        ds_train = data_utils.CDMDataset(stage='train',
                                        n_samples=self.args.n_samples,
                                        data_dir=self.args.data_dir)
        
        ds_val = data_utils.CDMDataset(stage='val',
                                        n_samples=self.args.n_samples,
                                        data_dir=self.args.data_dir)
        
        ds_test = data_utils.CDMDataset(stage='test',
                                        n_samples=self.args.n_samples,
                                        data_dir=self.args.data_dir)
        

        self.dl_train = DataLoader(ds_train, 
                                shuffle=True, 
                                batch_size=self.args.batch_size, 
                                collate_fn=data_utils.collate_fn_padding)


        self.dl_val = DataLoader(ds_val, 
                                 shuffle=False, 
                                 batch_size=self.args.batch_size,
                                 collate_fn=data_utils.collate_fn_padding)
        
        self.dl_test = DataLoader(ds_test, 
                                  shuffle=False, 
                                  batch_size=self.args.batch_size,
                                  collate_fn=data_utils.collate_fn_padding)        

    def training_step(self, batch):
        xb, yb, mkb = batch
        xb, yb, mkb = xb.to(self.device), yb.to(self.device), mkb.to(self.device)
        if self.args.model == 'transformer':
            pred = self.model(xb, ~mkb)
        else:
            pred = self.model(xb)
        return self.criterion(pred[mkb], yb[mkb]).mean()
    
    def validation_step(self, batch):
        xb, yb, mkb = batch
        xb, yb, mkb = xb.to(self.device), yb.to(self.device), mkb.to(self.device)
        if self.args.model == 'transformer':
            pred = self.model(xb, ~mkb)
        else:
            pred = self.model(xb)
        return self.criterion(pred[mkb], yb[mkb]).sum().cpu().item(), mkb.sum().cpu().item()

    def fit(self):
        for epoch in range(self.args.epochs):
            start = time.time()
            self.model.train()
            for batch in self.dl_train:
                self.optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                if self.args.clip is not None: 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()
            self.scheduler.step()
            time_train = time.time() - start

            with torch.no_grad():
                start = time.time()
                self.model.eval()
                train_loss, val_loss, test_loss = 0, 0, 0
                train_count, val_count, test_count = 0, 0, 0
                for batch in self.dl_train:
                    loss, count = self.validation_step(batch)
                    train_loss += loss
                    train_count += count
                for batch in self.dl_val:
                    loss, count = self.validation_step(batch)
                    val_loss += loss
                    val_count += count
                for batch in self.dl_test:
                    loss, count = self.validation_step(batch)
                    test_loss += loss
                    test_count += count
                for batch in self.dl_eval:
                    eval_loss, eval_count = self.validation_step(batch)
                time_val = time.time() - start

            self.log('epoch', epoch)
            self.log('loss/train', train_loss/train_count)
            self.log('loss/val', val_loss/val_count)
            self.log('loss/test', test_loss/test_count)
            self.log('loss/eval', eval_loss/eval_count)
            self.log('lr', self.scheduler.get_last_lr()[0])
            self.log('time/train', time_train)  
            self.log('time/val', time_val)
            self.log.print()
            if self.early_stopping(epoch, self.log.get('loss/val'), self.model): break
        print(f'Early stopping at epoch {self.early_stopping.best_epoch}', flush=True)
        self.log.print(self.early_stopping.best_epoch)

"""
Simulation Trainner for Simulation Data (Same Length)
"""
class SimulationTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
    
    def configure_criterions(self):
        self.criterion_1 = metrics.nll_loss
        self.criterion_2 = nn.MSELoss(reduction='none')

    def configure_dataloaders(self):
        ds_train = data_utils.SimulationDataset(stage='train',
                                            n_samples=self.args.n_samples,
                                            data_dir=self.args.data_dir)
        
        ds_val = data_utils.SimulationDataset(stage='val',
                                        n_samples=self.args.n_samples,
                                        data_dir=self.args.data_dir)
        
        ds_test = data_utils.SimulationDataset(stage='test',
                                        n_samples=self.args.n_samples,
                                        data_dir=self.args.data_dir)

        self.dl_train = DataLoader(ds_train, batch_size=self.args.batch_size, shuffle=True)
        self.dl_val = DataLoader(ds_val, batch_size=self.args.batch_size, shuffle=False)
        self.dl_test = DataLoader(ds_test, batch_size=self.args.batch_size, shuffle=False)

    def training_step(self, batch):
        x, y, _ = batch
        x, y = x.to(self.device), y.to(self.device)
        pred = self.model(x)
        loss = self.criterion_1(pred, y)
        return loss

    def validation_step(self, batch):
        x, y, h = batch
        x, y, h = x.to(self.device), y.to(self.device), h.to(self.device)
        out = self.model(x)
        nll = self.criterion_1(out, y).mean().item()
        mse = self.criterion_2(out, h).mean().item()
        return nll, mse
    
    def fit(self):
        for epoch in range(self.args.epochs):
            start = time.time()
            self.model.train()
            for batch in self.dl_train:
                loss = self.training_step(batch).mean()
                loss.backward()
                if self.args.clip is not None: 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.scheduler.step()
            time_train = time.time() - start

            with torch.no_grad():
                start = time.time()
                self.model.eval()
                nll_train, mse_train = zip(*[self.validation_step(batch) for batch in self.dl_train])
                nll_val, mse_val = zip(*[self.validation_step(batch) for batch in self.dl_val])
                nll_test, mse_test = zip(*[self.validation_step(batch) for batch in self.dl_test])
                time_val = time.time() - start

            self.log('epoch', epoch)
            self.log('loss/train', np.mean(nll_train))
            self.log('loss/val', np.mean(nll_val))
            self.log('loss/test', np.mean(nll_test))
            self.log('mse/train', np.mean(mse_train))
            self.log('mse/val', np.mean(mse_val))
            self.log('mse/test', np.mean(mse_test))
            self.log('lr', self.scheduler.get_last_lr()[0]) 
            self.log('time/train', time_train)  
            self.log('time/val', time_val)
            self.log.print()
            if self.early_stopping(epoch, self.log.get('loss/val'), self.model): break
        print(f'Early stopping at epoch {self.early_stopping.best_epoch}', flush=True)
        self.log.print(self.early_stopping.best_epoch)

"""
Local Trainner 
"""
class LocalTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)

    def configure_dataloaders(self):
        # Dataset
        self.ds_train = data_utils.CDMDataset(n_samples=self.args.n_samples,
                                            data_dir=self.args.data_dir)
        
        self.ds_val = data_utils.CDMDataset(stage='val',
                                        n_samples=self.args.n_samples,
                                        data_dir=self.args.data_dir)
        
        self.ds_test = data_utils.CDMDataset(stage='test',
                                        n_samples=self.args.n_samples,
                                        data_dir=self.args.data_dir)
        
        self.rics = self.ds_train.rics
        
    def configure_callbacks(self):
        # Create directory
        os.makedirs(self.args.out_dir, exist_ok=True) # root
        os.makedirs(f'{self.args.out_dir}/models', exist_ok=True) # model 

        # Save args
        self.args.n_params  = sum(p.numel() for p in self.model.parameters() if p.requires_grad) 
        json.dump(self.args.__dict__, open(f'{self.args.out_dir}/args.json', 'w'), indent=4) 

        # Set stdout
        sys.stdout = open(f'{self.args.out_dir}/log.txt', 'w')

    def fit(self):
        if self.args.resume_path is not None:
            files = os.listdir(self.args.resume_path)
            rics_trained = [f.split('_')[-1][:-3] for f in files]
        else:
            rics_trained = []

        rics_failed = []
        for i, ric in enumerate(self.rics):
            if ric in rics_trained: continue
            try:    
                self.fit_one(i, ric)
                rics_trained.append(ric)
            except Exception as e:
                # print error
                print(f'Failed: {ric}, {e}', flush=True)
                rics_failed.append(ric)
                continue
        print(f'Failed: {rics_failed}', flush=True)

    def fit_one(self, idx, ric):
        self.configure_seeds()
        self.configure_models()
        self.configure_optimizers()
        self.configure_schedulers()
        self.early_stopping = callbacks.EarlyStopping(out_path=f'{self.args.out_dir}/models/{idx}_{ric}.pt',
                                                      patience=self.args.patience, 
                                                      delta=self.args.delta)
        
        x_train, y_train = self.ds_train[idx]
        x_val, y_val = self.ds_val[idx]
        x_test, y_test = self.ds_test[idx]

        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        x_val, y_val = x_val.to(self.device), y_val.to(self.device)
        x_test, y_test = x_test.to(self.device), y_test.to(self.device)

        for epoch in range(self.args.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            if self.args.model == 'transformer':
                pred = self.model(x_train, None)
            else:
                pred = self.model(x_train)
            loss = self.criterion(pred, y_train).mean()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                self.model.eval()
                loss_val = self.criterion(self.model(x_val), y_val).mean()
                loss_test = self.criterion(self.model(x_test), y_test).mean()

            if epoch % 50 == 0:
                print(f'idx: {idx}, ric: {ric}, epoch: {epoch}, loss/train: {loss:.3f}, loss/val: {loss_val:.3f}, loss/test: {loss_test:.3f}', flush=True)
            if self.early_stopping(epoch, loss_val, self.model): 
                print(f'earlystop, idx: {idx}, ric: {ric}, epoch: {self.early_stopping.best_epoch}, loss/val: {self.early_stopping.best_loss:.3f}', flush=True)
                break
