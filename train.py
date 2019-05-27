from config import *
from Input_Pipeline_celeba import *
from network_utility import *
from utilities import *

from tqdm import tqdm
import numpy as np
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

prev_phase_iter = get_prev_phase_iter()
print('prev_phase_iter:', prev_phase_iter)
if __name__ == '__main__':

    tf.reset_default_graph()

    real_img, mask = build_input_pipline(batch_size, train_tfrecord_path)

    z = get_z(batch_size, latent_size)

    #build model here
    real_img = tf.reshape(real_img, [batch_size, output_img_h, output_img_w, 3])
    mask = tf.reshape(mask, [batch_size, output_img_h, output_img_w, 1])

    print(real_img, mask)
    model = prog_w_gan(z, real_img, mask, phase = phase, LAMBDA=10)

    merged = tf.summary.merge_all()

    g_alpha = 1e-12
    d_alpha = 1e-12

    run_metadata = tf.RunMetadata()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        train_writer = tf.summary.FileWriter('runs/{}/logs/phase_{}'.format(exp_name, phase), sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        print('Session Initiated')


        if use_smooth and phase > 2:
            model.loader.restore(sess, 'runs/{}/model/phase_{}/iteration_latest/model_latest.ckpt'.format(exp_name, phase-1, prev_phase_iter))
            model.d_smooth_loader.restore(sess, 'runs/{}/model/phase_{}/iteration_latest/model_latest.ckpt'.format(exp_name, phase-1, prev_phase_iter))

            print('***Phase{}:  Phase_{} weights loaded***'.format(phase, phase-1))
        elif phase >= 1 and not use_smooth and prev_phase_iter > 0:
            model.saver.restore(sess, 'runs/{}/model/phase_{}/iteration_latest/model_latest.ckpt'.format(exp_name, phase))
            print('***Phase{}:  Phase_{} weights loaded***'.format(phase, phase))

        per_iter_time = 0
        with tqdm(total=int(epoch_num * total_samples / batch_size-prev_phase_iter), unit='it') as pbar:
            train_start_time = time.time()

            for iter in range(prev_phase_iter,int(epoch_num * total_samples / batch_size)):
                iter_start_time = time.time()

                smooth_factors = {model.g_alpha: g_alpha, model.d_alpha: d_alpha}

                for critic_itr in range(n_critic-1):
                    sess.run([model.apply_d_grad], feed_dict=smooth_factors)

                _, _, g_loss, d_loss, summary = sess.run([model.apply_d_grad, model.apply_g_grad, model.g_loss, model.d_loss, merged], feed_dict=smooth_factors)

                g_alpha = np.clip(g_alpha + 5e-5, 0, 1)
                d_alpha = np.clip(d_alpha + 5e-5, 0, 1)

                iter_per_sec = 1/(time.time() - iter_start_time)
                train_writer.add_summary(summary, iter)

                if iter % 10 == 0: pbar.set_postfix({'it_ins/s':'{:4.2f}, d_loss:{}, g_loss:{}'.format(iter_per_sec, d_loss, g_loss)})
                pbar.update(1)

                if iter == 0:
                    real_img, fake_masks = sess.run([model.real_images, model.fake_masks], feed_dict=smooth_factors)
                    save_png(real_img[:16,:,:,:], [4 , 4], 'runs/{}/samples/phase_{}/real_sample.png'.format(exp_name, phase))
                    save_tiff(fake_masks[:16,:,:,:], [4, 4], 'runs/{}/samples/phase_{}/fake_mask_{}.tif'.format(exp_name, phase, iter))

                if iter % int(1000) == 0:
                    fake_img, fake_masks = sess.run([model.fake_images, model.fake_masks], feed_dict=smooth_factors)
                    save_png(fake_img[:16,:,:,:], [4,4], 'runs/{}/samples/phase_{}/fake_{}.png'.format(exp_name, phase, iter))
                    save_tiff(fake_masks[:16,:,:,:], [4,4], 'runs/{}/samples/phase_{}/fake_mask_{}.tif'.format(exp_name, phase, iter))

                if iter % 1000 == 0 and iter != 0:
                    root = 'runs/{}/model/phase_{}/iteration_{}/'.format(exp_name, phase, iter)
                    if not os.path.exists(root):
                        os.makedirs(root)
                    model.saver.save(sess, root + 'model_{}.ckpt'.format(iter))

                    root = 'runs/{}/model/phase_{}/iteration_latest/'.format(exp_name, phase)
                    model.saver.save(sess, root + 'model_latest.ckpt')