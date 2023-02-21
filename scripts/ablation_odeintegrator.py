"""
@file ablation_odeintegrator.py

Holds the cmd calls to train models across different ODE integrators automatically
"""
import os

dev = 0
num_epochs = 100

os.system(f"python main.py --exptype ablation_odeint_rk4_1     --integrator rk4 --integrator_params step_size=0.5   --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")
os.system(f"python main.py --exptype ablation_odeint_rk4_0.5   --integrator rk4 --integrator_params step_size=0.5   --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")
os.system(f"python main.py --exptype ablation_odeint_rk4_0.25  --integrator rk4 --integrator_params step_size=0.25  --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")
os.system(f"python main.py --exptype ablation_odeint_rk4_0.125 --integrator rk4 --integrator_params step_size=0.125 --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")

os.system(f"python main.py --exptype ablation_odeint_dopri5_500  --integrator dopri5 --integrator_params max_num_steps=500  --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")
os.system(f"python main.py --exptype ablation_odeint_dopri5_1000 --integrator dopri5 --integrator_params max_num_steps=1000 --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")
os.system(f"python main.py --exptype ablation_odeint_dopri5_2000 --integrator dopri5 --integrator_params max_num_steps=2000 --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")
os.system(f"python main.py --exptype ablation_odeint_dopri5_5000 --integrator dopri5 --integrator_params max_num_steps=5000 --num_epochs {num_epochs} --latent_dim 8 --num_hidden 128 --num_layers 3 --num_filt 8 --dev {dev}")
