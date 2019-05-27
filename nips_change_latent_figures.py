#from complete_runs.gcp_1.Input_Pipeline_celeba import *
from complete_runs.gcp_1.network_utility import *
from utilities import *
from Input_Pipeline_celeba import *
from tqdm import tqdm
import numpy as np
import os
import time
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

train_tfrecord_path = train_file_dict[size]


#save_folder = '/data/nobackup/'
save_folder = ''

if __name__ == '__main__':

    tf.reset_default_graph()

    real_img, mask = build_input_pipline(batch_size, train_tfrecord_path)

    z = get_z(batch_size, latent_size)

    #build model here
    real_img = tf.reshape(real_img, [batch_size, output_img_h, output_img_w, 3])
    mask = tf.reshape(mask, [batch_size, output_img_h, output_img_w, 1])

    model = prog_w_gan(z, real_img, mask, phase = phase, LAMBDA=10)

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
        print('complete_runs/gcp_2/model/model.ckpt')

        print('***Weights loaded***')

        per_iter_time = 0


        counter = 0

        for iter in tqdm(range(1,200)):
            iter_start_time = time.time()

            smooth_factors = {model.g_alpha: g_alpha, model.d_alpha: d_alpha}


            real_img_batch_l, mask_l= sess.run([model.real_images, model.fake_masks], feed_dict=smooth_factors)

            real_img = real_img_batch_l[0, :, :, :]
            real_img = (real_img * 255 / 2 + 255 / 2).clip(0, 255).astype(np.uint8)

            mask = mask_l[0, :, :, :]

            save_one_png(real_img, '{}nips_figures/change_latent/phase_{}/sample_{}/real.png'.format(save_folder, phase, iter))
            save_one_tiff(mask, '{}nips_figures/change_latent/phase_{}/sample_{}/mask.tif'.format(save_folder, phase, iter))

            counter = 0
            for i in range(100):
                fake_image_l = sess.run([model.fake_images], feed_dict={model.g_alpha: g_alpha, model.d_alpha: d_alpha, model.fake_masks:mask_l})


                fake_img = fake_image_l[0][0, :, :, :]
                fake_img = (fake_img * 255 / 2 + 255 / 2).clip(0, 255).astype(np.uint8)

                save_one_png(fake_img, '{}nips_figures/change_latent/phase_{}/sample_{}/fake/fake_{}.png'.format(save_folder, phase, iter, counter))
                counter += 1


