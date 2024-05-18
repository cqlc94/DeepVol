# Empirical scaling 
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 128 --n_samples 10
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 128 --n_samples 50
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 128 --n_samples 100
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 128 --n_samples 500
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 128 --n_samples 1000
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 128 --n_samples 5000
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 128 --n_samples 10000

python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'rnn' --n_layers 1 --d_hidden 128 --n_samples 10
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'rnn' --n_layers 1 --d_hidden 128 --n_samples 50
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'rnn' --n_layers 1 --d_hidden 128 --n_samples 100
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'rnn' --n_layers 1 --d_hidden 128 --n_samples 500
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'rnn' --n_layers 1 --d_hidden 128 --n_samples 1000
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'rnn' --n_layers 1 --d_hidden 128 --n_samples 5000
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'rnn' --n_layers 1 --d_hidden 128 --n_samples 10000

python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'gru' --n_layers 1 --d_hidden 128 --n_samples 10
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'gru' --n_layers 1 --d_hidden 128 --n_samples 50
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'gru' --n_layers 1 --d_hidden 128 --n_samples 100
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'gru' --n_layers 1 --d_hidden 128 --n_samples 500
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'gru' --n_layers 1 --d_hidden 128 --n_samples 1000
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'gru' --n_layers 1 --d_hidden 128 --n_samples 5000

python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'transformer' --n_layers 1 --d_hidden 128 --n_samples 10
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'transformer' --n_layers 1 --d_hidden 128 --n_samples 50
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'transformer' --n_layers 1 --d_hidden 128 --n_samples 100
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'transformer' --n_layers 1 --d_hidden 128 --n_samples 500
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'transformer' --n_layers 1 --d_hidden 128 --n_samples 1000
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'transformer' --n_layers 1 --d_hidden 128 --n_samples 5000
python -u run.py --exp_type empirical --dataset '18k_stock.pkl' --model 'transformer' --n_layers 1 --d_hidden 128 --n_samples 10000


# Simulation Scaling
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'lstm' --n_layers 2 --d_hidden 256 --n_samples 10
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'lstm' --n_layers 2 --d_hidden 256 --n_samples 100
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'lstm' --n_layers 2 --d_hidden 256 --n_samples 1000
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'lstm' --n_layers 2 --d_hidden 256 --n_samples 10000
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'lstm' --n_layers 2 --d_hidden 256 --n_samples 100000

python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'garch' --n_samples 10
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'garch' --n_samples 100
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'garch' --n_samples 1000
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'garch' --n_samples 10000
python -u run.py --exp_type simulation --dataset '110k_garch.pkl' --model 'garch' --n_samples 100000


# Local Model
python -u run.py --exp_type 'local' --dataset '18_stock.pkl' --model 'lstm' --n_layers 1 --d_hidden 32 
python -u run.py --exp_type 'local' --dataset '110k_garch.pkl' --model 'lstm' --n_layers 1 --d_hidden 32 