python3 -W ignore train.py --GPU 4 --phase 2 --smooth 1 --size 32 --epoch 40 --batch_size 128 --lr 0.001 --n_critic 1 --use_embedding 1 #16
python3 -W ignore train.py --GPU 4 --phase 2 --smooth 0 --size 32 --epoch 95 --batch_size 128 --lr 0.001 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 4 --phase 3 --smooth 1 --size 16 --epoch 20 --batch_size 64 --lr 0.001 --n_critic 1 --use_embedding 1 #32
python3 -W ignore train.py --GPU 4 --phase 3 --smooth 0 --size 16 --epoch 60 --batch_size 64 --lr 0.001 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 4 --phase 4 --smooth 1 --size 8 --epoch 10 --batch_size 32 --lr 0.001 --n_critic 1 --use_embedding 1 #64
python3 -W ignore train.py --GPU 4 --phase 4 --smooth 0 --size 8 --epoch 26 --batch_size 32  --lr 0.001 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 4 --phase 5 --smooth 1 --size 4 --epoch 10 --batch_size 32 --lr 0.0012 --n_critic 1 --use_embedding 1 #128
python3 -W ignore train.py --GPU 4 --phase 5 --smooth 0 --size 4 --epoch 26 --batch_size 32 --lr 0.0012 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 4 --phase 6 --smooth 1 --size 4 --epoch 10 --batch_size 32 --lr 0.0018 --n_critic 1 --use_embedding 1 #256
python3 -W ignore train.py --GPU 4 --phase 6 --smooth 0 --size 4 --epoch 26 --batch_size 32 --lr 0.0018 --n_critic 1 --use_embedding 1

python3 -W ignore train.py --GPU 4 --phase 7 --smooth 1 --size 2 --epoch 5 --batch_size 16 --lr 0.0018 --n_critic 1 --use_embedding 1 # 512
python3 -W ignore train.py --GPU 4 --phase 7 --smooth 0 --size 2 --epoch 30 --batch_size 16 --lr 0.00185 --n_critic 1 --use_embedding 1
