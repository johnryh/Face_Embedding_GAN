from config import *
import tensorflow as tf
import numpy as np
import os
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def activation_func(z, activation='leaky_relu'):
    if activation == 'leaky_relu':
        return tf.nn.leaky_relu(z)
    elif activation == 'relu':
        return tf.nn.relu(z)
    elif activation == 'selu':
        return tf.nn.selu(z)
    elif activation == 'tanh':
        return tf.nn.tanh(z)
    elif activation == 'sigmoid':
        return tf.nn.sigmoid(z)
    elif activation == 'linear':
        return z

    assert False, 'Activation Func "{}" not Found'.format(activation)


def PN(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)

    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))


def minibatch_stddev_layer(x, group_size=4):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape                                             # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[3], s[1], s[2]])   # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.

        return tf.concat([x, y], axis=3)                        # [NCHW]  Append as new fmap.


def dense(z, units, activation=None, name='Dense', gain=np.sqrt(2)/4, use_PN=False):
    with tf.variable_scope(name):
        with tf.device("/device:{}:0".format(controller)):
            assert len(z.shape) == 2, 'Input Dimension must be rank 2, but is rank {}'.format(len(z.shape))
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=gain)
            weights = get_weight([z.shape[1].value, units], gain, use_wscale=True)
            biases = tf.get_variable('bias', [units], initializer=initializer)

            y = tf.add(tf.matmul(z, weights), biases)

            if activation:
                y= activation_func(y, activation)

            if use_PN:
                y = PN(y)

            return y


def conv2d(input_vol, input_dim, num_kernal, scope, kernal_size=3, stride=1, activation='leaky_relu', padding='SAME', batch_norm=False, gain=np.sqrt(2), use_PN=False):
    with tf.variable_scope(scope):
        if isinstance(kernal_size, int):
            kernal_height = kernal_size
            kernal_width = kernal_size
        else:
            kernal_height = kernal_size[0]
            kernal_width = kernal_size[1]

        with tf.device("/device:{}:0".format(controller)):
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=gain)
            weights = get_weight([kernal_height, kernal_width, int(input_vol.shape[-1]), int(num_kernal)], gain, use_wscale=True)
            biases = tf.get_variable('bias', [int(num_kernal)], initializer=initializer)

        conv = tf.add(tf.nn.conv2d(input_vol, weights, [1, stride, stride, 1], padding=padding), biases)

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=True)

        out = activation_func(conv, activation)

        if use_PN:
            out = PN(out)

        return out


def conv2d_transpose(z, kernel_num, kernel_size, stride, padding='SAME', isTrain=True, activation='leaky_relu', name='kernel', batch_norm=False, factor=2, use_PN=False, gain=np.sqrt(2)):
    with tf.variable_scope(name):
        with tf.device("/device:{}:0".format(controller)):
            initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32, factor=np.sqrt(2))
            #kernel = tf.get_variable('weights', [kernel_size, kernel_size, int(kernel_num), z.shape[-1].value], initializer=initializer)
            weights = get_weight([kernel_size, kernel_size, int(kernel_num), z.shape[-1].value], gain, use_wscale=True, fan_in=(kernel_size**2)*z.shape[1].value)
            biases = tf.get_variable('bias', [int(kernel_num)], initializer=initializer)

            shape = tf.constant([z.shape[0].value, z.shape[1].value*factor, z.shape[2].value*factor, kernel_num])

        conv = tf.add(tf.nn.conv2d_transpose(z, weights, shape, strides=[1, stride[0], stride[1], 1], padding=padding), biases)

        if batch_norm:
            conv = tf.layers.batch_normalization(conv, training=isTrain)

        out = activation_func(conv, activation)

        if use_PN:
            out = PN(out)

        return out


def input_projection(x, num_base_features, batch_norm=False):
    with tf.variable_scope('Input_Projection'):
        if batch_norm:
            x = tf.layers.batch_normalization(x, training=True)

        latent = dense(x, units=4 * 4 * num_base_features, activation='leaky_relu', name='Dense_1', gain=np.sqrt(2)/4, use_PN=False)
        latent = tf.reshape(latent, [-1, 4, 4, num_base_features])

        latent = conv2d(latent, 0, num_kernal=g_max_num_features, kernal_size=3, stride=1, padding='SAME', scope='Projection_Conv', use_PN=True)

        return latent


def rgb_projection(fetures, input_dim, name='output_projection', is_smooth=False):
    with tf.variable_scope(name):
        #fetures = conv2d(fetures, input_dim, num_kernal=max(int(int(fetures.shape[-1])/4), 16), kernal_size=3, stride=1, padding='SAME', scope='reduced_features', activation='leaky_relu', use_PN=True)
        output = conv2d(fetures, input_dim, num_kernal=3, kernal_size=1, stride=1, padding='SAME', scope='to_rgb', activation=to_rgb_activation, gain=1)

        #output = tf.clip_by_value(output, 0, 1)
    return output


def from_rgb_mask(image, mask, num_features, name):
    with tf.variable_scope(name):
        mask_features = conv2d(mask, 1, 3, 'mask_conv', kernal_size=3, stride=1, padding='SAME')
        x = tf.concat([image, mask_features], axis=3)
        features = conv2d(x, -1, num_features, 'to_feature_space', kernal_size=3, stride=1, padding='SAME')


        return features


def score_projection(features_volume, num_featrues_list, name='score_projection'):
    with tf.variable_scope(name):
        reg1 = conv2d(features_volume, -1, num_featrues_list[0], 'conv_reg1_{}'.format(1), kernal_size=(3, 3),stride=1, padding='SAME')

        with tf.variable_scope('Flatten'):
            flatten_features = tf.reshape(reg1, [int(batch_size/num_gpus), -1])

        latent_feature = dense(flatten_features, units=num_featrues_list[1], name='Dense_0', activation='linear', gain=1)
        score = dense(latent_feature, units=1, name='Dense', activation='linear', gain=1)

        return score


def mask_projection_path_more_features(fake_masks):
    #num_mask_features = {0:8, 1:8, 2:16, 3:16, 4:32, 5:32, 6:32, 7:32, 8:32}
    num_mask_features = {0:32, 1:32, 2:32, 3:16, 4:16, 5:8, 6:8, 7:8, 8:8}

    use_PN = False
    mask_features = [fake_masks]
    out_features = [fake_masks]

    target_size = 16
    if len(mask_features) == 1:
        curr_block_phase = int(math.log(int(mask_features[-1].shape[1]) / 4, 2))
        with tf.variable_scope('mask_to_fetures'):
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=8, scope='mask_fetures', kernal_size=8, stride=1, use_PN=use_PN, padding='SAME'))
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=num_mask_features[curr_block_phase], scope='mask_fetures_l', kernal_size=8, stride=1, use_PN=use_PN, padding='SAME'))

            num_concat_features = 8 if curr_block_phase >= 4 else 8

            out_features.append(mask_features[-1][:, :, :, :num_concat_features])


    while mask_features[-1].shape[1] > target_size:
        curr_block_phase = int(math.log(int(mask_features[-1].shape[1]) / 4, 2))-1
        with tf.variable_scope('mask_{}'.format(curr_block_phase-1)):
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=num_mask_features[curr_block_phase], scope='mask_conv_{}'.format(curr_block_phase-1), kernal_size=4, stride=1, use_PN=use_PN, padding='SAME'))
            mask_features.append(conv2d(mask_features[-1], mask_features[-1].shape[3], num_kernal=num_mask_features[curr_block_phase], scope='mask_conv_{}_l'.format(curr_block_phase-1), kernal_size=4, stride=2, use_PN=use_PN, padding='SAME'))

            num_concat_features = 8 if  curr_block_phase >= 4 else 8

            out_features.append(mask_features[-1][:, :, :, :num_concat_features])

    if use_embedding:
        with tf.variable_scope('Mask_Embedding'):
            mask_features.append(dense(tf.reshape(mask_features[-1], [int(batch_size/num_gpus), target_size * target_size * mask_features[-1].shape[-1]]), embedding_latent_0_size, activation='linear', name='embedding_latent_0_size', gain=1))
            #mask_features.append(dense(mask_features[-1], embedding_latent_1_size, activation='linear', name='embedding_latent_1_size', gain=1))
            mask_features.append(dense(mask_features[-1], embedding_size, name='embedding_out', gain=np.sqrt(2)/4))
            out_features.append(mask_features[-1])


    return out_features, mask_features


def generator(z, phase_count, num_base_features, num_feature_decay, alpha, isTrain=True, mask=None):

    # alpha 0 to 1, the weight of newly added layer features to the gray projection
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):

        if mask is not None: mask_features, all_mask_features = mask_projection_path_more_features(mask)
        mask_features = mask_features[::-1]
        all_mask_features = all_mask_features[::-1]
        # build image synthesizer path
        with tf.variable_scope('Input_Level'):
            if mask is not None and use_embedding: z = tf.concat([mask_features[0], z], axis=1)

            latent = input_projection(z, num_base_features=num_base_features)

        num_features = [num_base_features]
        blocks = [latent]
        for i in range(phase_count):
            with tf.variable_scope('block_{}'.format(i)):
                if i < 3: # is first block
                    num_features.append(int(num_features[-1]))
                else:
                    num_features.append(np.clip(int(num_features[-1] * num_feature_decay), 16, g_max_num_features))
                kernel_size = 3 if i < 5 else 4
                blocks.append(conv2d_transpose(blocks[-1], num_features[-1], kernel_size, (2, 2), padding='SAME', isTrain=isTrain, name='c2d_trs_p{}'.format(i+1), use_PN=conv2d_transpose_use_pn))
                if mask is not None and i > 0: blocks[-1] = tf.concat([blocks[-1], mask_features[i + use_embedding - 1]], axis=3)
                print('block:',i)

                blocks.append(conv2d(blocks[-1], num_features[-1], num_features[-1], 'conv_p{}_l'.format(i+1), kernal_size=3, stride=1, padding='SAME', use_PN=conv2d_transpose_use_pn))
        if phase > 1 and use_smooth:
            # add short connection for smoothing
            with tf.variable_scope('smooth_connection'):
                features_double = tf.image.resize_images(blocks[-3], [4*2**(phase_count), 4*2**(phase_count)], method=tf.image.ResizeMethod.BILINEAR)
                short_connection = rgb_projection(features_double,num_features[-2], name='smooth_projection', is_smooth=True)

        out = rgb_projection(blocks[-1], num_features[-1])

        with tf.variable_scope('rgb_output'):
            if phase > 1 and use_smooth:
                rgb_images = alpha*out + (1-alpha)*short_connection
            else:
                rgb_images = out

        return rgb_images, all_mask_features


def dicsriminator(image, phase_count, num_base_features, num_feature_multiplier, max_num_features, alpha, scope='Discriminator', mask=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        num_features = [np.clip(int(num_base_features * num_feature_multiplier**(9-phase_count)), 0, max_num_features)]

        blocks = [from_rgb_mask(image=image, mask=mask, num_features=num_features[-1], name='from_rgb_mask')] if mask is not None else [conv2d(image, image.shape[3], num_features[-1], 'from_rgb', kernal_size=3, stride=1, padding='SAME')]

        if phase_count > 1 and use_smooth:
            # add short connection for smoothing
            with tf.variable_scope('smooth_connection'):
                rgb_half = tf.nn.avg_pool(image, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='rgb_half')
                mask_half = tf.nn.max_pool(mask, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='mask_half')

                smooth_features = from_rgb_mask(image=rgb_half, mask=mask_half, num_features=np.clip(int(num_features[-1] * num_feature_multiplier), 0, max_num_features), name='from_rgb_smooth') if mask is not None else conv2d(image, image.shape[3], num_features[-1], 'from_rgb_smooth', kernal_size=3, stride=1, padding='SAME')

        for i in range(1, phase_count+1):
            num_features.append(np.clip(int(num_features[-1] * num_feature_multiplier), 16, max_num_features))
            if i == 2 and use_smooth:
                with tf.variable_scope('Feature_Interp'):
                    features = alpha * blocks[-1] + (1 - alpha) * smooth_features
                with tf.variable_scope('p{}'.format(phase_count-i+1)):
                    blocks.append(conv2d(features, num_features[-2], num_features[-2], 'conv_p{}'.format(phase_count-i+1), kernal_size=3, stride=1, padding='SAME'))
                    blocks.append(conv2d(blocks[-1], num_features[-1], num_features[-1], 'conv_p{}_l'.format(phase_count-i+1), kernal_size=3, stride=2, padding='SAME'))
            else:
                with tf.variable_scope('p{}'.format(phase_count-i+1)):
                    blocks.append(conv2d(blocks[-1], num_features[-2], num_features[-2], 'conv_p{}'.format(phase_count-i+1), kernal_size=3, stride=1, padding='SAME'))
                    blocks.append(conv2d(blocks[-1], num_features[-1], num_features[-1], 'conv_p{}_l'.format(phase_count-i+1), kernal_size=3, stride=2, padding='SAME'))

        blocks.append(minibatch_stddev_layer(blocks[-1]))
        out = score_projection(blocks[-1], [num_features[-2], num_features[-1]])

        return out, blocks


def get_pc_var_list():
    var_list = []
    for v in tf.global_variables():
        if ('/score_projection/' in v.name and 'conv' not in v.name) or '/Input_Projection/' in v.name:
            pass
        else:
            var_list.append(v)
            #print('pc:', v.name)

    return var_list


def get_var_list_by_phase(curr_phase):
    var_list = []
    for v in tf.global_variables():
        for key_word in ['c2d_trs_p', 'conv_p']:
            if key_word in v.name:
                phase = int(v.name.split(key_word)[-1][0])
                if phase < curr_phase :
                    var_list.append(v)

        if 'mask_conv_' in v.name and 'rgb' not in v.name:
            phase = int(v.name.split('mask_conv_')[-1][0])
            if phase <= curr_phase-3:
                var_list.append(v)

        elif 'Input_Projection' in v.name:
            var_list.append(v)

        elif 'score_projection' in v.name:
            var_list.append(v)

        elif 'Mask_Embedding' in v.name:
            var_list.append(v)


    return var_list


def get_d_smooth_loader_var_list():
    smooth_var_dict = {}
    for v in tf.global_variables():
        if 'smooth' in v.name and 'Discriminator' in v.name:
            smooth_var_dict[v.name.replace('/smooth_connection/from_rgb_smooth/','/from_rgb_mask/').replace(':0', '')] = v

        if 'smooth_projection' in v.name and 'Generator' in v.name:
            smooth_var_dict[v.name.replace('/smooth_connection/smooth_projection/', '/output_projection/').replace(':0', '')] = v


    return smooth_var_dict


def get_z(batch_size, z_length):
    with tf.variable_scope('z'):
        z = tf.random_normal(shape=[batch_size, z_length], mean=0, stddev=1, name='random_z')

    return z


class prog_w_gan():

    def __init__(self, latent, real_img, mask, phase, LAMBDA):
        self.initialize_optimizer()
        self.build_model(latent, real_img,  mask, phase=phase, LAMBDA=LAMBDA)
        self.model_stats()

    def initialize_optimizer(self):
        with tf.variable_scope('GAN_Optimizer'):
            with tf.variable_scope('D_Optim'):
                self.d_optim = tf.train.AdamOptimizer(learning_rate=d_ln_rate, beta1=0.0, beta2=0.99, epsilon=1e-8, name='d_optim') # was 0.5, 0.9
            with tf.variable_scope('G_Optim'):
                self.g_optim = tf.train.AdamOptimizer(learning_rate=g_ln_rate, beta1=0.0, beta2=0.99, epsilon=1e-8, name='g_optim')
        print('Solver Configured')

    def build_model(self, latent, real_img, mask, phase, LAMBDA):
        latent_split = tf.split(latent, num_gpus, name='latent_split') if num_gpus > 1 else [latent]
        real_img_split = tf.split(real_img, num_gpus, name='real_img_split') if num_gpus > 1 else [real_img]
        mask_split = tf.split(mask, num_gpus, name='mask_split') if num_gpus > 1 else [mask]


        tower_real_score=[]; tower_fake_score=[]
        tower_g_loss=[]; tower_d_loss=[]
        tower_fake_image=[]
        tower_g_grads = [];tower_d_grads = [];
        self.g_alpha = tf.placeholder(dtype=tf.float32, shape=())
        self.d_alpha = tf.placeholder(dtype=tf.float32, shape=())
        self.real_images = real_img
        self.fake_masks = mask
        for gpu_id in range(num_gpus):
            with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                with tf.variable_scope('GAN') as GAN_scope:

                    fake_images, self.mask_features = generator(latent_split[gpu_id], phase_count=phase, num_base_features=g_max_num_features, num_feature_decay=0.5, alpha=self.g_alpha, mask=mask_split[gpu_id])
                    fake_score, self.fake_blocks = dicsriminator(fake_images, phase_count=phase, num_base_features= 8, num_feature_multiplier=2, max_num_features=d_max_num_features, alpha=self.d_alpha, mask=mask_split[gpu_id])
                    real_score, self.real_blocks = dicsriminator(real_img_split[gpu_id], phase_count=phase, num_base_features= 8, num_feature_multiplier=2, max_num_features=d_max_num_features, alpha=self.d_alpha, mask=mask_split[gpu_id])

                with tf.variable_scope('Gan_Loss'):
                    with tf.variable_scope('Tower_G_loss'):
                        g_loss = tf.reduce_mean(fake_score)
                    with tf.variable_scope('Tower_D_Loss'):
                        d_loss = -tf.reduce_mean(fake_score) + tf.reduce_mean(real_score)
                    with tf.variable_scope('GP_Calculation'):
                        w_alpha = tf.random_uniform([int(batch_size/num_gpus),1,1,1], 0.0, 1.0)
                        with tf.variable_scope('differences'):
                            img_differences = fake_images - real_img_split[gpu_id]

                        with tf.variable_scope('interpolates'):
                            interpolates = real_img_split[gpu_id] + (w_alpha * img_differences)
                        with tf.variable_scope(GAN_scope):
                            gradients = tf.gradients(dicsriminator(interpolates, phase_count=phase, num_base_features=8, num_feature_multiplier=2, max_num_features=d_max_num_features, alpha=self.d_alpha, mask=mask_split[gpu_id])[0], [interpolates])[0]
                            #gradients = tf.gradients(dicsriminator(interpolates, phase_count=phase, num_base_features=8, num_feature_multiplier=2, max_num_features=d_max_num_features, alpha=self.d_alpha, mask=mask_split[gpu_id])[0], [interpolates, mask_split[gpu_id]])[0]

                        with tf.variable_scope('Gradient_Penalty'):
                            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                            d_loss += LAMBDA * gradient_penalty
                        with tf.variable_scope('D_Drift_Penalty'):
                            d_drift_penalty = e_drift * tf.reduce_mean(tf.square(real_score))

                        d_loss += d_drift_penalty

                    if gpu_id == 0:
                        # trainable variables for each network
                        T_vars = tf.trainable_variables()

                        D_vars = [var for var in T_vars if 'Discriminator' in var.name]
                        G_vars = [var for var in T_vars if 'Generator' in var.name]
                        print('var_list initiated')

                        tf.summary.scalar('g_loss/fake_score', g_loss)
                        tf.summary.scalar('d_loss', d_loss)
                        tf.summary.scalar('real_score', tf.reduce_mean(real_score))
                        tf.summary.scalar('alpha', self.g_alpha)
                        tf.summary.scalar('gradient_penalty', gradient_penalty)
                        tf.summary.scalar('d_drift_penalty', d_drift_penalty)

                        self.saver = tf.train.Saver(name='P{}_Saver'.format(phase), max_to_keep=None)
                        self.loader = tf.train.Saver(name='P{}_Loader'.format(phase),var_list=get_var_list_by_phase(phase)) if phase > 1 else None
                        if use_smooth: self.d_smooth_loader = tf.train.Saver(name='P{}_d_Smooth_Loader'.format(phase),var_list=get_d_smooth_loader_var_list()) if phase > 1 else None

                    with tf.variable_scope('Compute_Optim_Gradients'):
                        tower_g_grads.append(self.g_optim.compute_gradients(g_loss, var_list=G_vars))
                        tower_d_grads.append(self.d_optim.compute_gradients(d_loss, var_list=D_vars))

                tower_d_loss.append(d_loss)
                tower_g_loss.append(g_loss)
                tower_fake_score.append(fake_score)
                tower_real_score.append(real_score)
                tower_fake_image.append(fake_images)

        with tf.variable_scope('Sync_Point'):
            self.g_loss = tf.reduce_mean(tower_g_loss, axis=0, name='g_loss')
            self.d_loss = tf.reduce_mean(tower_d_loss, axis=0, name='d_loss')
            self.fake_score = tf.concat(tower_fake_score, axis=0)
            self.real_score = tf.concat(tower_real_score, axis=0)
            self.fake_images = tf.concat(tower_fake_image, axis=0)

        with tf.variable_scope('GAN_Solver'):

            with tf.variable_scope('Apply_Optim_Gradients'), tf.device("/device:{}:1".format(controller)):
                self.g_grads = self.average_gradients(tower_g_grads)
                self.d_grads = self.average_gradients(tower_d_grads)

                self.apply_g_grad = self.g_optim.apply_gradients(self.g_grads, name='Apply_G_Grads')
                self.apply_d_grad = self.d_optim.apply_gradients(self.d_grads, name='Apply_D_Grads')

    def average_gradients(self, tower_grads):
        with tf.variable_scope('Average_Gradients'):

            average_grads = []
            for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                grads = [g for g, _ in grad_and_vars]
                grad = tf.reduce_mean(grads, 0)

                # Keep in mind that the Variables are redundant because they are shared
                # across towers. So .. we will just return the first tower's pointer to
                # the Variable.
                v = grad_and_vars[0][1]
                grad_and_var = (grad, v)

                average_grads.append(grad_and_var)
            return average_grads

    def model_stats(self):
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GAN/Discriminator'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Discriminator Total parameters:{}M'.format(total_parameters / 1e6))

        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='GAN/Generator'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Generator Total parameters:{}M'.format(total_parameters / 1e6))

        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('Total Total parameters:{}M'.format(total_parameters / 1e6))
