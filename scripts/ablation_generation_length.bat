python train.py --generation_len 1 --exptype node_pendulum_1gen --generation_varying False
python test.py  --ckpt_path experiments/node_pendulum_1gen/node/version_1/ --training_len 1

python train.py --generation_len 2 --exptype node_pendulum_2gen --generation_varying False
python test.py  --ckpt_path experiments/node_pendulum_2gen/node/version_1/ --training_len 2

python train.py --generation_len 3 --exptype node_pendulum_3gen --generation_varying False
python test.py  --ckpt_path experiments/node_pendulum_3gen/node/version_1/ --training_len 3

python train.py --generation_len 5 --exptype node_pendulum_5gen --generation_varying False
python test.py  --ckpt_path experiments/node_pendulum_5gen/node/version_1/ --training_len 5

python train.py --generation_len 10 --exptype node_pendulum_10gen --generation_varying False
python test.py  --ckpt_path experiments/node_pendulum_10gen/node/version_1/ --training_len 10

python train.py --generation_len 20 --exptype node_pendulum_20gen --generation_varying False
python test.py  --ckpt_path experiments/node_pendulum_20gen/node/version_1/ --training_len 20
