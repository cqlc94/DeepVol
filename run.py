import datetime
import argparse
from lib.trainner import EmpiricalTrainer, SimulationTrainer, LocalTrainer
import time

""" Args """
"""
Empirical:
Simulation: change exp_type, dataset, n_samples and batch size
Local: change exp_type, epochs, patience, lr, lr_min, T_max
"""
# Exp
parser = argparse.ArgumentParser()
parser.add_argument('--exp_type', type=str, default='empirical')
parser.add_argument('--dataset', type=str, default='18k_stock.pkl')
parser.add_argument('--resume_path', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
# Training
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--delta', type=float, default=1e-8)
parser.add_argument('--device', type=str, default='cuda')
# Data
parser.add_argument('--n_samples', type=int, default=int(1e4))
parser.add_argument('--context_len', type=int, default=None)
# Model
parser.add_argument('--model', type=str, default='lstm')
parser.add_argument('--d_in', type=int, default=1)
parser.add_argument('--d_hidden', type=int, default=16)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.2)
# Model Transformer
parser.add_argument('--nhead', type=int, default=2)
parser.add_argument('--d_ff', type=int, default=128)
# Optimization
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--clip', type=float, default=1)
# Scheduler
parser.add_argument('--scheduler', type=str, default='constant')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_min', type=float, default=1e-4)
parser.add_argument('--T_max', type=float, default=200)
parser.add_argument('--gamma', type=float, default=0.99)
# Add-on
args = parser.parse_args()
args.data_dir = f'data/training/{args.dataset}'
args.out_dir = f'checkpoints/{args.exp_type}/{args.model}_{args.n_samples}_{args.d_hidden}_{datetime.datetime.now().strftime("%m%d%H%M")}'
""" Log """
trainer_classes = {'empirical': EmpiricalTrainer, 'simulation': SimulationTrainer, 'local': LocalTrainer}
trainer = trainer_classes[args.exp_type](args)
trainer.fit()