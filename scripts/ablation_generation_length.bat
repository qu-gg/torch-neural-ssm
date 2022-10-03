python train.py --generation_len 1 --exptype node_pendulum_1gen
python test.py  --generation_len 1 --ckpt_path experiments/node_pendulum_1gen/node/version_1/

python train.py --generation_len 2 --exptype node_pendulum_2gen
python test.py  --generation_len 2 --ckpt_path experiments/node_pendulum_2gen/node/version_1/

python train.py --generation_len 3 --exptype node_pendulum_3gen
python test.py  --generation_len 3 --ckpt_path experiments/node_pendulum_3gen/node/version_1/

python train.py --generation_len 5 --exptype node_pendulum_5gen
python test.py  --generation_len 5 --ckpt_path experiments/node_pendulum_5gen/node/version_1/

python train.py --generation_len 10 --exptype node_pendulum_10gen
python test.py  --generation_len 10 --ckpt_path experiments/node_pendulum_10gen/node/version_1/

python train.py --generation_len 20 --exptype node_pendulum_20gen
python test.py  --generation_len 20 --ckpt_path experiments/node_pendulum_20gen/node/version_1/
