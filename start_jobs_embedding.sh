python3 -W ignore train.py --GPU 3 --phase 1 --smooth 1 --size 32 --epoch 50 --batch_size 192 --lr 0.001 --n_critic 1 --use_embedding 1 # 8
python3 -W ignore train.py --GPU 3 --phase 1 --smooth 0 --size 32 --epoch 125 --batch_size 192 --lr 0.001 --n_critic 1 --use_embedding 1 

python3 -W ignore train.py --GPU 3 --phase 2 --smooth 1 --size 32 --epoch 25 --batch_size 96 --lr 0.001 --n_critic 1 --use_embedding 1 #16
python3 -W ignore train.py --GPU 3 --phase 2 --smooth 0 --size 32 --epoch 60 --batch_size 96 --lr 0.001 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 3 --phase 3 --smooth 1 --size 8 --epoch 13 --batch_size 48 --lr 0.001 --n_critic 1 --use_embedding 1 #32
python3 -W ignore train.py --GPU 3 --phase 3 --smooth 0 --size 8 --epoch 40 --batch_size 48 --lr 0.001 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 3 --phase 4 --smooth 1 --size 16 --epoch 8 --batch_size 24 --lr 0.0008 --n_critic 1 --use_embedding 1 #64
python3 -W ignore train.py --GPU 3 --phase 4 --smooth 0 --size 16 --epoch 20 --batch_size 24 --lr 0.0008 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 3 --phase 5 --smooth 1 --size 8 --epoch 4 --batch_size 12 --lr 0.001 --n_critic 1 --use_embedding 1 # 128
python3 -W ignore train.py --GPU 3 --phase 5 --smooth 0 --size 8 --epoch 15 --batch_size 12 --lr 0.001 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 3 --phase 6 --smooth 1 --size 4 --epoch 4 --batch_size 8 --lr 0.0018 --n_critic 1 --use_embedding 1 #  256
python3 -W ignore train.py --GPU 3 --phase 6 --smooth 0 --size 4 --epoch 10 --batch_size 8 --lr 0.0018 --n_critic 1 --use_embedding 1
