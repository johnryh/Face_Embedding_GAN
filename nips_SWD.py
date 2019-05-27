from Input_Pipeline_celeba import *
from utilities import *

from complete_runs.gcp_1.network_utility import *

from tqdm import tqdm
import numpy as np
import os
import time
from matplotlib import pyplot as plt
from sliced_wasserstein_impl import sliced_wasserstein_distance

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

train_tfrecord_path = train_file_dict[size]

save_folder = '/data/nobackup/'
group_size = 20

if __name__ == '__main__':

    tf.reset_default_graph()

    real_img, mask = build_input_pipline(batch_size, train_tfrecord_path)
    print('Currenttrainig file:', train_tfrecord_path)

    z = get_z(batch_size, latent_size)

    #build model here
    real_img = tf.reshape(real_img, [batch_size, output_img_h, output_img_w, 3])
    mask = tf.reshape(mask, [batch_size, output_img_h, output_img_w, 1])

    model = prog_w_gan(z, real_img, mask, phase = phase, LAMBDA=10)

    real_img_holder = tf.placeholder(dtype=tf.float32, shape=([group_size*batch_size, 512, 512, 3]))
    fake_img_holder = tf.placeholder(dtype=tf.float32, shape=([group_size*batch_size, 512, 512, 3]))

    with tf.device("/device:{}:0".format('CPU')):
        SWD = sliced_wasserstein_distance(real_img_holder, fake_img_holder, resolution_min=16, patch_size=7, random_sampling_count=1, patches_per_image=128)

    g_alpha = 1e-12
    d_alpha = 1e-12

    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('Session Initiated')

        model.saver.restore(sess, 'complete_runs/gcp_1/model/model.ckpt')
        print('***Weights loaded***')

        per_iter_time = 0

        real_swd_512_list = []
        real_swd_256_list = []
        real_swd_128_list = []
        real_swd_64_list = []
        real_swd_32_list = []
        real_swd_16_list = []

        fake_swd_512_list = []
        fake_swd_256_list = []
        fake_swd_128_list = []
        fake_swd_64_list = []
        fake_swd_32_list = []
        fake_swd_16_list = []
        for i in range(10):

            real_img_list = []
            fake_img_list = []
            #for iter in tqdm(range(1, int(16384/batch_size))):
            for iter in tqdm(range(0, group_size)):

                iter_start_time = time.time()

                smooth_factors = {model.g_alpha: g_alpha, model.d_alpha: d_alpha}

                real_img_batch, fake_img_batch = sess.run([model.real_images, model.fake_images], feed_dict=smooth_factors)
                real_img_list.append(real_img_batch)
                fake_img_list.append(fake_img_batch)


            real_img_list = np.reshape(real_img_list, [-1, 512,512,3])
            fake_img_list = np.reshape(fake_img_list, [-1, 512,512,3])
            print(real_img_list.shape)

            curr_SWD = sess.run([SWD], feed_dict={real_img_holder:real_img_list, fake_img_holder:fake_img_list})

            swd_512 = curr_SWD[0][0][0]*1000
            swd_256 = curr_SWD[0][1][0]*1000
            swd_128 = curr_SWD[0][2][0]*1000
            swd_64 = curr_SWD[0][3][0]*1000
            swd_32 = curr_SWD[0][4][0]*1000
            swd_16 = curr_SWD[0][5][0]*1000
            print('real --- 512: {:.2f}, 256: {:.2f}, 128: {:.2f}, 64: {:.2f}, 32: {:.2f}, 16: {:.2f}'.format(swd_512, swd_256, swd_128, swd_64, swd_32, swd_16))
            real_swd_512_list.append(swd_512)
            real_swd_256_list.append(swd_256)
            real_swd_128_list.append(swd_128)
            real_swd_64_list.append(swd_64)
            real_swd_32_list.append(swd_32)
            real_swd_16_list.append(swd_16)


            swd_512 = curr_SWD[0][0][1]*1000
            swd_256 = curr_SWD[0][1][1]*1000
            swd_128 = curr_SWD[0][2][1]*1000
            swd_64 = curr_SWD[0][3][1]*1000
            swd_32 = curr_SWD[0][4][1]*1000
            swd_16 = curr_SWD[0][5][1]*1000
            print('fake --- 512: {:.2f}, 256: {:.2f}, 128: {:.2f}, 64: {:.2f}, 32: {:.2f}, 16: {:.2f} '.format(swd_512, swd_256, swd_128, swd_64, swd_32, swd_16))
            fake_swd_512_list.append(swd_512)
            fake_swd_256_list.append(swd_256)
            fake_swd_128_list.append(swd_128)
            fake_swd_64_list.append(swd_64)
            fake_swd_32_list.append(swd_32)
            fake_swd_16_list.append(swd_16)

            print('real_avg --- 512: {:.2f}, 256: {:.2f}, 128: {:.2f}, 64: {:.2f}, 32: {:.2f}, 16: {:.2f}'.format(np.mean(real_swd_512_list), np.mean(real_swd_256_list), np.mean(real_swd_128_list), np.mean(real_swd_64_list), np.mean(real_swd_32_list), np.mean(real_swd_16_list)))
            print('fake_avg --- 512: {:.2f}, 256: {:.2f}, 128: {:.2f}, 64: {:.2f}, 32: {:.2f}, 16: {:.2f}\n'.format(np.mean(fake_swd_512_list), np.mean(fake_swd_256_list), np.mean(fake_swd_128_list), np.mean(fake_swd_64_list), np.mean(fake_swd_32_list), np.mean(fake_swd_16_list)))

