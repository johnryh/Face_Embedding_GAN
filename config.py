import argparse
import os

# config phase

parser = argparse.ArgumentParser('')
parser.add_argument('--GPU', dest='num_gpus', required=True)
parser.add_argument('--phase', dest='phase', required=True)
parser.add_argument('--smooth', dest='use_smooth', required=True)
parser.add_argument('--size', dest='size', required=True)
parser.add_argument('--epoch', dest='epoch_num', required=True)
parser.add_argument('--batch_size', dest='batch_size', required=True)
parser.add_argument('--lr', dest='lr', required=True)
parser.add_argument('--n_critic', dest='n_critic', required=True)
parser.add_argument('--use_embedding', dest='use_embedding', required=True)
print('config')
controller = 'CPU'

args = parser.parse_args()
num_gpus = int(args.num_gpus)
phase = int(args.phase)
use_smooth = int(args.use_smooth)
size = int(args.size)
epoch_num = int(args.epoch_num)
batch_size = int(args.batch_size)
lr = float(args.lr)
n_critic = int(args.n_critic)
use_embedding = int(args.use_embedding)

loading_img_h = int(1024/size)
loading_img_w = loading_img_h
print('loading:',[loading_img_h, loading_img_h])

output_img_h = int(8 * 2 ** (phase - 1))
output_img_w = output_img_h

print('output:',[output_img_h, output_img_w])

data_folder = '/data/nobackup/celebrity_hq_mask_bilinear_tf'
train_file_dict = {1: '{}/celeba-r10.tfrecords'.format(data_folder), # 1024
                   2: '{}/celeba-r09.tfrecords'.format(data_folder), # 512
                   4: '{}/celeba-r08.tfrecords'.format(data_folder), # 256
                   8: '{}/celeba-r07.tfrecords'.format(data_folder), # 128
                   16: '{}/celeba-r06.tfrecords'.format(data_folder), # 64
                   32: '{}/celeba-r05.tfrecords'.format(data_folder), # 32
                   64: '{}/celeba-r05.tfrecords'.format(data_folder), # 16
                   128: '{}/celeba-r05.tfrecords'.format(data_folder)} # 8

train_tfrecord_path = train_file_dict[size]

png_encode = True if 'png' in train_tfrecord_path else False

# model parameters
embedding_size = 48 if use_embedding else 0
embedding_latent_0_size = 256
embedding_latent_1_size = 128

latent_size = 512 - embedding_size
g_max_num_features = 512
d_max_num_features = 512
g_ln_rate = lr
d_ln_rate = lr
e_drift = 0.0001
FLIP_RATE = 0.5

conv2d_transpose_use_pn = True
to_rgb_activation = 'linear'
exp_name = 'face_mask_no_embedding_final'

total_samples = 7e4
