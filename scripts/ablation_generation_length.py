"""
@file ablation_generation_length.py

Holds the cmd calls to train models across different generation lengths used in training
"""
import os

os.system("python3 main.py --generation_len 1 --exptype node_pendulum_1gen --generation_varying False")
os.system("python3 test.py  --ckpt_path experiments/node_pendulum_1gen/node/version_1/ --training_len 1")

os.system("python3 main.py --generation_len 2 --exptype node_pendulum_2gen --generation_varying False")
os.system("python3 test.py  --ckpt_path experiments/node_pendulum_2gen/node/version_1/ --training_len 2")

os.system("python3 main.py --generation_len 3 --exptype node_pendulum_3gen --generation_varying False")
os.system("python3 test.py  --ckpt_path experiments/node_pendulum_3gen/node/version_1/ --training_len 3")

os.system("python3 main.py --generation_len 5 --exptype node_pendulum_5gen --generation_varying False")
os.system("python3 test.py  --ckpt_path experiments/node_pendulum_5gen/node/version_1/ --training_len 5")

os.system("python3 main.py --generation_len 10 --exptype node_pendulum_10gen --generation_varying False")
os.system("python3 test.py  --ckpt_path experiments/node_pendulum_10gen/node/version_1/ --training_len 10")

os.system("python3 main.py --generation_len 20 --exptype node_pendulum_20gen --generation_varying False")
os.system("python3 test.py  --ckpt_path experiments/node_pendulum_20gen/node/version_1/ --training_len 20")
