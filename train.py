
"""
Auto-Encoder models
"""
import os
import sys
import logging

import numpy as np
import tensorflow.compat.v1 as tf
from math import ceil, pi

import utils
from priors import init_prior
from sampling_functions import sample_pz, traversals, interpolations, grid, shift, rotate
from plot_functions import save_train, save_test
from plot_functions import plot_critic_pretrain_loss
from plot_functions import plot_interpolation, plot_cost_shift, plot_rec_shift, plot_embedded_shift
import models
# from networks import theta_discriminator
from wgan import wgan, wgan_v2
from loss_functions import wae_ground_cost
from datahandler import datashapes
from fid.fid import calculate_frechet_distance

import pdb
import typing
from datahandler import DataHandler


class Run(object):

    def __init__(self, opts: dict, data: DataHandler) -> None:

        logging.error('Building the Tensorflow Graph')
        self.opts = opts

        # --- Data
        assert opts['dataset'] in datashapes, 'Unknown dataset.'
        self.data = data

        # --- Initialize prior parameters
        self.pz_mean, self.pz_Sigma = init_prior(opts)

        # --- Placeholders
        self.add_ph()

        # --- Instantiate Model
        self.model: typing.Optional[models.Model] = None
        if opts['model'] == 'BetaVAE':
            self.model = models.BetaVAE(opts, self.pz_mean, self.pz_Sigma)
        elif opts['model'] == 'WAE':
            self.model = models.WAE(opts, self.pz_mean, self.pz_Sigma)
        else:
            raise NotImplementedError()

        # --- Define Objectives
        self.cost, self.loss_reg, self.intensities_reg, self.critic_reg = self.model.loss(
                                    inputs=self.data.next_element,
                                    is_training=self.is_training)
        # rec loss
        self.loss_rec = self.cost + opts['gamma'] * self.intensities_reg
        # wae obj
        self.objective = self.loss_rec + self.beta * self.loss_reg
        # critic obj
        if self.critic_reg is not None:
            self.critic_objective = -self.loss_rec + opts['lambda']*self.critic_reg
        else:
            self.critic_objective = tf.constant(0., tf.float32)

        # --- Critic loss for pretraining
        if self.opts['cost']=='wgan' and self.opts['pretrain_critic']:
            critic_loss, _, critic_reg = wgan(self.opts, self.data.next_element,
                                        tf.random.shuffle(self.data.next_element),
                                        is_training=self.is_training,
                                        reuse=True)
            self.critic_pretrain_loss = tf.reduce_mean(critic_loss - opts['lambda']*critic_reg)


        # --- encode & decode pass
        self.z_samples, self.z_mean, self.z_sigma, self.recon_x, _, _ =\
            self.model.forward_pass(inputs=self.data.next_element,
                                    is_training=self.is_training,
                                    reuse=True)

        # --- Pen Encoded Sigma &  stats
        Sigma_tr = tf.reduce_mean(self.z_sigma, axis=-1)
        Smean, Svar = tf.nn.moments(Sigma_tr, axes=[0])
        self.encSigmas_stats = tf.stack([Smean, Svar], axis=-1)
        if self.opts['pen_enc_sigma'] and self.opts['encoder'] == 'gauss':
            pen_enc_sigma = self.opts['beta_pen_enc_sigma'] * tf.reduce_mean(
                tf.reduce_sum(tf.abs(tf.math.log(self.z_sigma)), axis=-1))
            self.objective+= pen_enc_sigma

        # --- MSE
        self.mse = self.model.MSE(self.data.next_element, self.recon_x)

        # --- FID score
        if self.opts['fid']:
            self.inception_graph = tf.Graph()
            self.inception_sess = tf.Session(graph=self.inception_graph)
            with self.inception_graph.as_default():
                self.create_inception_graph()
            self.inception_layer = self._get_inception_layer()

        # --- Get batchnorm ops for training only
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # --- encode & decode pass for vizu
        self.encoded, self.encoded_mean, _, self.decoded, _, _ =\
            self.model.forward_pass(inputs=self.inputs_img1,
                                    is_training=self.is_training,
                                    reuse=True)

        # --- Sampling
        self.generated = self.model.sample_x_from_prior(noise=self.pz_samples)

        # --- Rec cost obs vs reconstruction
        cost, _, intensities_reg, _ = self.model.loss(self.inputs_img1,
                                    is_training=self.is_training,
                                    reuse=True)
        # rec loss
        self.rec_cost = cost# + opts['gamma'] * intensities_reg
        # MSE
        self.rec_mse = self.model.MSE(self.inputs_img1, self.decoded)

        # --- Rec cost for given obs
        cost, intensities_reg, _ = wae_ground_cost(self.opts,
                                    self.inputs_img2,
                                    self.inputs_img1,
                                    is_training=self.is_training,
                                    reuse=True)
        self.ground_cost = tf.reduce_mean(cost) # + opts['gamma']*intensities_reg)
        self.ground_mse = self.model.MSE(self.inputs_img2, self.inputs_img1)

        # --- Optimizers, savers, etc
        self.add_optimizers()

        # --- Init iteratorssess, saver and load trained weights if needed, else init variables
        self.sess = tf.Session()
        self.train_handle, self.test_handle = self.data.init_iterator(self.sess)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.initializer = tf.global_variables_initializer()
        self.sess.graph.finalize()


    def add_ph(self) -> None:
        self.lr_decay = tf.placeholder(tf.float32, name='rate_decay_ph')
        self.is_training = tf.placeholder(tf.bool, name='is_training_ph')
        self.pz_samples = tf.placeholder(tf.float32,
                                         [None] + [self.opts['zdim'],],
                                         name='noise_ph')
        self.inputs_img1 = tf.placeholder(tf.float32,
                                         [None] + self.data.data_shape,
                                         name='point1_ph')
        self.inputs_img2 = tf.placeholder(tf.float32,
                                         [None] + self.data.data_shape,
                                         name='point2_ph')
        self.beta = tf.placeholder(tf.float32, name='beta_ph')

    def compute_blurriness(self) -> tf.Tensor:
        images = self.inputs_img
        # First convert to greyscale
        if self.data.data_shape[-1] > 1:
            # We have RGB
            images = tf.image.rgb_to_grayscale(images)
        # Next convolve with the Laplace filter
        lap_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        lap_filter = lap_filter.reshape([3, 3, 1, 1])
        conv = tf.nn.conv2d(images, lap_filter, strides=[1, 1, 1, 1],
                                                padding='VALID')
        _, lapvar = tf.nn.moments(conv, axes=[1, 2, 3])
        return lapvar

    def create_inception_graph(self) -> None:
        inception_model = 'classify_image_graph_def.pb'
        inception_path = os.path.join('fid', inception_model)
        # Create inception graph
        with tf.gfile.FastGFile(inception_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString( f.read())
            _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')

    def _get_inception_layer(self) -> typing.Any:
        # Get inception activation layer (and reshape for batching)
        layername = 'FID_Inception_Net/pool_3:0'
        pool3 = self.inception_sess.graph.get_tensor_by_name(layername)
        ops_pool3 = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops_pool3):
            for o in op.outputs:
                shape = o.get_shape()
                if shape._dims != []:
                  shape = [s.value for s in shape]
                  new_shape: typing.List[int] = []
                  for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                      new_shape.append(None)
                    else:
                      new_shape.append(s)
                  o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
        return pool3

    def optimizer(self, lr: float, decay=1.) -> tf.train.Optimizer:
        opts = self.opts
        lr *= decay
        if opts['optimizer'] == 'sgd':
            return tf.train.GradientDescentOptimizer(lr)
        elif opts['optimizer'] == 'adam':
            return tf.train.AdamOptimizer(lr, beta1=opts['adam_beta1'], beta2=opts['adam_beta2'])
        else:
            assert False, 'Unknown optimizer.'

    def adam_discr_optimizer(self, lr=1e-4, beta1=0.9, beta2=0.999) -> tf.train.AdamOptimizer:
        return tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)

    def RMSProp_discr_optimizer(self, lr=5e-5) -> tf.train.RMSPropOptimizer:
        return tf.train.RMSPropOptimizer(lr)

    def add_optimizers(self) -> None:
        opts = self.opts
        # Encoder/decoder optimizer
        lr = opts['lr']
        opt = self.optimizer(lr, self.lr_decay)
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='decoder')
        with tf.control_dependencies(self.extra_update_ops):
            self.opt = opt.minimize(loss=self.objective, var_list=encoder_vars + decoder_vars)

        # wgan/wgan-gp critic optimizer
        if self.opts['cost'][:4]=='wgan':
            # critic_opt = self.RMSProp_discr_optimizer()
            critic_opt = self.adam_discr_optimizer(lr=1e-4)
            # critic_opt = self.adam_discr_optimizer(lr=1e-4,beta1=0.5,beta2=0.9)
            critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='w1_critic')
            with tf.control_dependencies(self.extra_update_ops):
                self.critic_opt = critic_opt.minimize(loss=self.critic_objective, var_list=critic_vars)
            # weights clipping
            clip_ops = []
            for var in critic_vars:
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var,tf.clip_by_value(var,
                                            clip_bounds[0], clip_bounds[1])))
            self.clip_critic_weights = tf.group(*clip_ops)
            # pretraining
            if self.opts['pretrain_critic']:
                with tf.control_dependencies(self.extra_update_ops):
                    self.critic_pretrain_opt = critic_opt.minimize(loss=-self.critic_pretrain_loss, var_list=critic_vars)


    def train(self, WEIGHTS_FILE: typing.Optional[str]=None) -> None:
        """
        Train top-down model with chosen method
        """
        logging.error('\nTraining {}'.format(self.opts['model']))
        exp_dir = self.opts['exp_dir']

        # - Set up for training
        train_size = self.data.train_size
        logging.error('\nTrain size: {}, trBatch num.: {}, Ite. num: {}'.format(
                                            train_size,
                                            int(train_size/self.opts['batch_size']),
                                            self.opts['it_num']))
        npics = self.opts['plot_num_pics']
        fixed_noise = sample_pz(self.opts, self.pz_mean, self.pz_Sigma, npics)
        # anchors_ids = np.random.choice(npics, 5, replace=True)
        anchors_ids = [0, 4, 6, 12, 39]

        # - Init all monitoring variables
        Loss, Loss_test, = [], []
        Losses_monit, Losses_monit_test = [], []
        MSE, MSE_test = [], []
        FID_rec, FID_gen = [], []
        if self.opts['vizu_encSigma']:
            enc_Sigmas = []
        # - Init decay lr and beta
        decay = 1.
        decay_rate = 0.9
        fix_decay_steps = 25000
        wait = 0
        batches_num = self.data.train_size//self.opts['batch_size']
        ada_decay_steps = batches_num
        wait_beta = 0

        # - Testing iterations number
        test_it_num = int(10000 / self.opts['batch_size'])

        # - Load trained model or init variables
        if self.opts['use_trained']:
            if WEIGHTS_FILE is None:
                    raise Exception("No model/weights provided")
            else:
                if not tf.gfile.IsDirectory(opts['exp_dir']):
                    raise Exception("model doesn't exist")
                WEIGHTS_PATH = os.path.join(opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
                if not tf.gfile.Exists(WEIGHTS_FILE+".meta"):
                    raise Exception("weights file doesn't exist")
                self.saver.restore(self.sess, WEIGHTS_FILE)
        else:
            self.sess.run(self.initializer)

        # - Critic pretraining
        if self.opts['cost'][:4]=='wgan' and self.opts['pretrain_critic']:
            logging.error('Pretraining Critic')
            pretrain_loss = []
            for i in range(self.opts['pretrain_critic_nit']):
                _, critic_pretrain_loss = self.sess.run(
                                    [self.critic_pretrain_opt,
                                    self.critic_pretrain_loss],
                                    feed_dict={self.data.handle: self.train_handle,
                                               self.is_training: True})
                if i%int(self.opts['pretrain_critic_nit']/200)==0:
                    pretrain_loss.append(critic_pretrain_loss)
            plot_critic_pretrain_loss(self.opts, pretrain_loss,
                                exp_dir,'critic_pretrain_loss.png')
            logging.error('Pretraining done.')


        # - Training
        for it in range(self.opts['it_num']):
            # Saver
            if it > 0 and it % self.opts['save_every'] == 0:
                self.saver.save(self.sess,
                                os.path.join(exp_dir, 'checkpoints', 'trained-wae'),
                                global_step=it)
            #####  TRAINING LOOP #####
            it += 1
            # # training theta_discriminator if needed
            # if self.opts['cost']=='sw' and (self.opts['sw_proj_type']=='max-sw' or self.opts['sw_proj_type']=='max-gsw'):
            #     if (it-1)%self.opts['d_updt_freq']==0:
            #         for i in range(self.opts['d_updt_it']):
            #             _ = self.sess.run(self.theta_discr_opt, feed_dict={
            #                                 self.data.handle: self.train_handle,
            #                                 self.is_training: True})
            # training w1 critic if needed
            if self.opts['cost'][:4]=='wgan':
                if (it-1)%self.opts['d_updt_freq']==0:
                    for i in range(self.opts['d_updt_it']):
                        _ = self.sess.run(self.critic_opt,
                                            feed_dict={self.data.handle: self.train_handle,
                                                       self.is_training: True})
            # training
            _ = self.sess.run(self.opt, feed_dict={
                                self.data.handle: self.train_handle,
                                self.lr_decay: decay,
                                self.beta: self.opts['beta'],
                                self.is_training: True})

            ##### TESTING LOOP #####
            if it % self.opts['evaluate_every'] == 0:
                # logging.error('\nIteration {}/{}'.format(it, self.opts['it_num']))
                feed_dict={self.data.handle: self.train_handle,
                                self.beta: self.opts['beta'],
                                self.is_training: False}
                losses = self.sess.run([self.objective,
                                            self.loss_rec,
                                            self.loss_reg,
                                            self.cost,
                                            self.intensities_reg,
                                            self.critic_objective,
                                            self.mse],
                                            feed_dict=feed_dict)
                Loss.append(losses[0])
                Losses_monit.append(losses[1:-1])
                MSE.append(losses[-1])

                # Encoded Sigma
                if self.opts['vizu_encSigma']:
                    enc_sigmastats = self.sess.run(self.encSigmas_stats,
                                            feed_dict=feed_dict)
                    enc_Sigmas.append(enc_sigmastats)

                loss, monitoring, mse = 0., np.zeros(5), 0.
                for it_ in range(test_it_num):
                    test_feed_dict={self.data.handle: self.test_handle,
                                    self.beta: self.opts['beta'],
                                    self.is_training: False}
                    losses = self.sess.run([self.objective,
                                            self.loss_rec,
                                            self.loss_reg,
                                            self.cost,
                                            self.intensities_reg,
                                            self.critic_objective,
                                            self.mse],
                                            feed_dict=test_feed_dict)
                    loss += losses[0] / test_it_num
                    monitoring += np.array(losses[1:-1]) / test_it_num
                    mse += losses[-1] / test_it_num
                Loss_test.append(loss)
                Losses_monit_test.append(monitoring)
                MSE_test.append(mse)

                # Printing various loss values
                logging.error('')
                debug_str = 'IT: %d/%d, ' % (it, self.opts['it_num'])
                logging.error(debug_str)
                debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1],Loss_test[-1])
                logging.error(debug_str)
                debug_str = 'REC=%.3f, TEST REC=%.3f, MSE=%10.3e, TEST MSE=%10.3e'  % (
                                            Losses_monit[-1][0] + self.opts['gamma']*Losses_monit[-1][3],
                                            Losses_monit_test[-1][0]+ self.opts['gamma']*Losses_monit_test[-1][3],
                                            MSE[-1],
                                            MSE_test[-1])
                logging.error(debug_str)
                if self.opts['model'] == 'BetaVAE':
                    debug_str = 'beta*KL=%10.3e, beta*TEST KL=%10.3e'  % (
                                            self.opts['beta']*Losses_monit[-1][1],
                                            self.opts['beta']*Losses_monit_test[-1][1])
                    logging.error(debug_str)
                elif self.opts['model'] == 'WAE':
                    debug_str = 'beta*MMD=%10.3e, beta*TEST MMD=%10.3e' % (
                                            self.opts['beta']*Losses_monit[-1][1],
                                            self.opts['beta']*Losses_monit_test[-1][1])
                    logging.error(debug_str)
                else:
                    raise NotImplementedError('Model type not recognised')

                if self.opts['fid']:
                    FID_rec.append(self.fid_score(fid_inputs='reconstruction'))
                    FID_gen.append(self.fid_score(fid_inputs='samples'))

            ##### Vizu LOOP #####
            if it % self.opts['print_every'] == 0:
                # - Encode, decode and sample
                # Auto-encoding test images & samples generated by the model
                [reconstructions_vizu, latents_vizu, generations] = self.sess.run(
                                            [self.decoded, self.encoded, self.generated],
                                            feed_dict={self.inputs_img1: self.data.data_vizu,
                                                       self.pz_samples: fixed_noise,
                                                       self.is_training: False})
                # Auto-encoding training images
                inputs_tr = []
                n_next_element = ceil(npics / self.opts['batch_size'])
                for _ in range(n_next_element):
                    inputs_tr.append(self.sess.run(self.data.next_element, feed_dict={self.data.handle: self.train_handle}))
                inputs_tr = np.concatenate(inputs_tr,axis=0)[:npics]
                reconstructions_train = self.sess.run(self.decoded,
                                            feed_dict={self.inputs_img1: inputs_tr,
                                                       self.is_training: False})
                # Saving plots
                save_train(self.opts,
                          inputs_tr, self.data.data_vizu,                       # train/vizu images
                          reconstructions_train, reconstructions_vizu,          # reconstructions
                          generations,                                          # model samples
                          Loss, Loss_test,                                      # loss
                          Losses_monit, Losses_monit_test,                      # losses split
                          MSE, MSE_test,                                        # mse
                          FID_rec, FID_gen,                                     # FID
                          exp_dir,                                              # working directory
                          'res_it%07d.png' % (it))                              # filename

                # - Latent interpolation
                # Encoded sigma
                if self.opts['vizu_encSigma'] and it > 1:
                    plot_encSigma(self.opts,
                                  enc_Sigmas,
                                  exp_dir,
                                  'encSigma_e%04d_mb%05d.png' % (epoch, buff*batches_num+it+1))

                # Encode anchors points and interpolate
                if self.opts['vizu_interpolation']:
                    num_steps = 15

                    enc_var = np.ones(self.opts['zdim'])
                    # create linespace
                    enc_interpolation = linespace(self.opts, num_steps,  # shape: [nanchors, zdim, nsteps, zdim]
                                            anchors=latents_vizu[anchors_ids],
                                            std=enc_var)
                    enc_interpolation = np.reshape(enc_interpolation, [-1, self.opts['zdim']])
                    # reconstructing
                    dec_interpolation = self.sess.run(self.generated,
                                            feed_dict={self.pz_samples: enc_interpolation,
                                                       self.is_training: False})
                    inter_anchors = np.reshape(dec_interpolation, [-1, self.opts['zdim'], num_steps]+self.data.data_shape)
                    plot_interpolation(self.opts, inter_anchors,
                                            exp_dir, 'inter_it%07d.png' % (it))

                # - Non linear proj is gsw
                if self.opts['cost']=='sw' and self.opts['sw_proj_type']=='max-gsw':
                    proj = self.sess.run(self.projections,
                                            feed_dict={self.inputs_img1: self.data.data_vizu})
                    proj = np.reshape(proj, (-1, self.data.data_shape[0], self.data.data_shape[1], self.opts['sw_proj_num']))
                    plot_projected(self.opts, self.data.data_vizu, proj,
                                            exp_dir, 'proj_it%07d.png' % (it))


            # - Update learning rate if necessary and it
            if self.opts['lr_decay']:
                # decaying every fix_decay_steps
                if it % fix_decay_steps == 0:
                    decay = decay_rate ** (int(it / fix_decay_steps))
                    logging.error('Reduction in lr: %f\n' % decay)
                # If no significant progress was made in the last epoch
                # then decrease the learning rate.
                # if np.mean(Loss_rec[-ada_decay_steps:]) < np.mean(Loss_rec[-5*ada_decay_steps:]) - np.var(Loss_rec[-5*ada_decay_steps:]):
                #     wait = 0
                # else:
                #     wait += 1
                # if wait > ada_decay_steps:
                #     decay = decay_rate ** (int(it / ada_decay_steps))
                #     logging.error('Reduction in lr: %f\n' % decay)
                #     wait = 0


            # - Update regularizer if necessary
            if self.opts['beta_schedule'] == 'adaptive':
                if it >= .0 and len(Loss_rec) > 0:
                    if wait_beta > 1000 * batches_num + 1:
                        # opts['beta'] = list(2*np.array(opts['beta']))
                        self.opts['beta'][-1] = 2*self.opts['beta'][-1]
                        wae_beta = self.opts['beta']
                        logging.error('beta updated to %s\n' % wae_beta)
                        print('')
                        wait_beta = 0
                    else:
                        wait_beta += 1

            # - logging
            if (it)%50000 ==0 :
                logging.error('')
                logging.error('Train it.: {}/{}'.format(it,self.opts['it_num']))

        # - Save the final model
        if self.opts['save_final'] and it > 0:
            self.saver.save(self.sess, os.path.join(exp_dir,
                                                'checkpoints',
                                                'trained-{}-final'.format(self.opts['model'])),
                                                global_step=it)

        # - Finale losses & scores
        feed_dict={self.data.handle: self.train_handle,
                        self.beta: self.opts['beta'],
                        self.is_training: False}
        losses = self.sess.run([self.objective,
                                    self.loss_rec,
                                    self.loss_reg,
                                    self.cost,
                                    self.intensities_reg,
                                    self.critic_objective,
                                    self.mse],
                                    feed_dict=feed_dict)
        Loss.append(losses[0])
        Losses_monit.append(losses[1:-1])
        MSE.append(losses[-1])
        # Test losses
        loss, monitoring, mse = 0., np.zeros(5), 0.
        for it_ in range(test_it_num):
            test_feed_dict={self.data.handle: self.test_handle,
                            self.beta: self.opts['beta'],
                            self.is_training: False}
            losses = self.sess.run([self.objective,
                                    self.loss_rec,
                                    self.loss_reg,
                                    self.cost,
                                    self.intensities_reg,
                                    self.critic_objective,
                                    self.mse],
                                    feed_dict=test_feed_dict)
            loss += losses[0] / test_it_num
            monitoring += np.array(losses[1:-1]) / test_it_num
            mse += losses[-1] / test_it_num
        Loss_test.append(loss)
        Losses_monit_test.append(monitoring)
        MSE_test.append(mse)

        # Printing various loss values
        logging.error('')
        logging.error('Training done.')
        debug_str = 'TRAIN LOSS=%.3f, TEST LOSS=%.3f' % (Loss[-1], Loss_test[-1])
        logging.error(debug_str)
        debug_str = 'REC=%.3f, TEST REC=%.3f, MSE=%10.3e, TEST MSE=%10.3e'  % (
                                    Losses_monit[-1][0] + self.opts['gamma']*Losses_monit[-1][3],
                                    Losses_monit_test[-1][0]+ self.opts['gamma']*Losses_monit_test[-1][3],
                                    MSE[-1],
                                    MSE_test[-1])
        logging.error(debug_str)
        if self.opts['model'] == 'BetaVAE':
            debug_str = 'beta*KL=%10.3e, beta*TEST KL=%10.3e'  % (
                                    self.opts['beta']*Losses_monit[-1][1],
                                    self.opts['beta']*Losses_monit_test[-1][1])
            logging.error(debug_str)
        elif self.opts['model'] == 'WAE':
            debug_str = 'beta*MMD=%10.3e, beta*TEST MMD=%10.3e' % (
                                    self.opts['beta']*Losses_monit[-1][1],
                                    self.opts['beta']*Losses_monit_test[-1][1])
            logging.error(debug_str)
        else:
            raise NotImplementedError('Model type not recognised')

        # - save training data
        if self.opts['save_train_data']:
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            utils.create_dir(save_path)
            name = 'res_train_final'
            np.savez(os.path.join(save_path, name),
                    loss=np.array(Loss), loss_test=np.array(Loss_test),
                    losses=np.array(Losses_monit), losses_test=np.array(Losses_monit_test),
                    mse=np.array(MSE), mse_test=np.array(MSE_test))

        if self.opts['fid']:
            FID_rec.append(self.fid_score(fid_inputs='reconstruction'))
            FID_gen.append(self.fid_score(fid_inputs='samples'))
            data_dir = 'train_data'
            save_path = os.path.join(exp_dir, data_dir)
            if not tf.io.gfile.isdir(save_path):
                utils.create_dir(save_path)
            name = 'fid_final'
            np.savez(os.path.join(save_path, name),
                    fid_rec=np.array(FID_rec), fid_gen=np.array(FID_gen))

    def plot(self, WEIGHTS_FILE: typing.Optional[str]=None) -> None:
        """
        Plots reconstructions, latent transversals and model samples
        """

        opts = self.opts

        # - Load trained model
        if WEIGHTS_FILE is None:
                raise Exception("No model/weights provided")
        else:
            if not tf.gfile.IsDirectory(opts['exp_dir']):
                raise Exception("model doesn't exist")
            WEIGHTS_PATH = os.path.join(opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
            if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_PATH)

        # - Set up
        im_shape = datashapes[opts['dataset']]
        num_pics = opts['plot_num_pics']
        num_steps = 20
        enc_var = np.ones(opts['zdim'])
        fixed_noise = sample_pz(opts, self.pz_mean, self.pz_Sigma, num_pics)
        anchors_ids = np.arange(0,num_pics,int(num_pics/12))
        # anchors_ids = [0, 4, 6, 12, 24, 35] # 39, 53, 60, 73, 89]
        # anchors_ids = list(np.arange(0,100,5))

        # - Auto-encoding test images & samples generated by the model
        [reconstructions_vizu, latents_vizu, generations] = self.sess.run(
                                    [self.decoded, self.encoded, self.generated],
                                    feed_dict={self.inputs_img1: self.data.data_vizu,
                                               self.pz_samples: fixed_noise,
                                               self.is_training: False})

        # - Visualization of embeddedings
        num_encoded = 500
        if opts['dataset'][-5:]=='mnist':
            idx = np.random.choice(np.arange(len(self.data.all_labels)), size=num_encoded, replace=False)
            data_mnist = self.data.all_data[idx]
            label_mnist = self.data.all_labels[idx]
        if opts['dataset'] == 'shifted_mnist':
            batch = np.zeros([num_encoded,] + self.data.data_shape)
            labels = np.zeros(label_mnist.shape, dtype=int)
            # shift data
            for n, obs in enumerate(data_mnist):
                # padding mnist img
                paddings = [[2,2], [2,2], [0,0]]
                obs = np.pad(obs, paddings, mode='constant', constant_values=0.)
                shape = obs.shape
                # create img
                img = np.zeros(self.data.data_shape)
                # sample cluster pos
                i = np.random.binomial(1, 0.5)
                pos_x = i*int(3*shape[0]/8)
                pos_y = i*int(3*shape[1]/8)
                # sample shift
                shift_x = np.random.randint(0, int(shape[0]/8))
                shift_y = np.random.randint(0, int(shape[1]/8))
                # place digit
                img[pos_x+shift_x:shape[0]+pos_x+shift_x, pos_y+shift_y:shape[1]+pos_y+shift_y] = obs
                batch[n] = img
                labels[n] = label_mnist[n] + 2*i
        elif opts['dataset'] == 'shifted_3pos_mnist':
            batch = np.zeros([num_encoded,] + self.data.data_shape)
            labels = np.zeros(label_mnist.shape, dtype=int)
            # shift data
            for n, obs in enumerate(data_mnist):
                # padding mnist img
                #paddings = [[2,2], [2,2], [0,0]]
                #obs = np.pad(obs, paddings, mode='constant', constant_values=0.)
                shape = obs.shape
                # create img
                img = np.zeros(self.data.data_shape)
                # sample cluster pos
                i = np.random.randint(3)
#                pos_x = i*int(self.data.data_shape[0]/4)
#                pos_y = i*int(self.data.data_shape[1]/4)
                if i==0:
                    pos_x = 8; pos_y = 8
                elif i==1:
                    pos_x = 16; pos_y = 16
                else:
                    pos_x = 24; pos_y = 24

                # place digit
                img[pos_x:shape[0]+pos_x, pos_y:shape[1]+pos_y] = obs
                batch[n] = img
                labels[n] = label_mnist[n] + i
        elif opts['dataset'] == 'rotated_mnist':
            # rotate the data
            # padding mnist img
            paddings = [[0,0], [2,2], [2,2], [0,0]]
            x_pad = np.pad(data_mnist, paddings, mode='constant', constant_values=0.)
            # rot image with 0.5 prob
            choice = np.random.randint(0,2,num_encoded).reshape([num_encoded,1,1,1])
            batch = np.where(choice==0, x_pad, np.rot90(x_pad,axes=(1,2)))
            labels = (label_mnist / 5).astype(np.int64) + 2*choice.reshape([num_encoded,])
            labels = label_mnist
        elif opts['dataset'] == 'gmm':
            batch = np.zeros([num_encoded,]+self.data.data_shape)
            labels = np.zeros([num_encoded,], dtype=int)
            logits_shape = [int(datashapes['gmm'][0]/2),int(datashapes['gmm'][1]/2),datashapes['gmm'][2]]
            for n in range(num_encoded):
                # choose mixture
                mu = np.zeros(logits_shape)
                choice = np.random.randint(0,2)
                mu[3*choice:3*choice+3,3*choice:6*choice+3] = np.ones((3,3,1))
                mu[1+3*choice,1+3*choice] = [1.5]
                # sample cat. logits
                logits = np.random.normal(mu,.1,size=logits_shape).reshape((-1))
                p = np.exp(logits) / np.sum(np.exp(logits))
                a = np.arange(np.prod(logits_shape))
                # sample pixel idx
                idx = np.random.choice(a,size=1,p=p)[0]
                i = int(idx / 6.)
                j = idx % 6
                # generate obs
                x = np.zeros(datashapes['gmm'])
                x[2*i:2*i+2,2*i:2*i+2] = np.ones((2,2,1))
                batch[n] = x
                labels[n] = choice
        else:
            assert False, 'Unknown {} dataset'.format(opts['dataset'])

        # encode
        encoded = self.sess.run(self.encoded, feed_dict={
                                    self.inputs_img1: batch,
                                    self.is_training: False})

        # - Rec, samples, embeddded
        save_test(opts, self.data.data_vizu, reconstructions_vizu,
                                    generations,
                                    encoded, labels,
                                    opts['exp_dir'])

        """
        # - Latent traversal
        # create linespace
        enc_interpolation = traversals(latents_vizu[anchors_ids],  # shape: [nanchors, nsteps, zdim]
                                    nsteps=num_steps,
                                    std=enc_var)
        enc_interpolation = np.reshape(enc_interpolation, [-1, opts['zdim']])
        # reconstructing
        dec_interpolation = self.sess.run(self.generated,
                                    feed_dict={self.pz_samples: enc_interpolation,
                                               self.is_training: False})
        inter_anchors = np.reshape(dec_interpolation, [-1, num_steps]+self.data.data_shape)
        plot_interpolation(self.opts, inter_anchors, opts['exp_dir'],
                                    'latent_traversal.png',
                                    train=False)
        """

        # - latent grid interpolation
        num_interpolation = 20
        grid_interpolation = grid(num_interpolation, opts['zdim'])
        grid_interpolation = np.reshape(grid_interpolation, [-1, opts['zdim']])
        # reconstructing
        obs_interpolation = self.sess.run(self.generated,
                                    feed_dict={self.pz_samples: grid_interpolation,
                                               self.is_training: False})
        obs_interpolation = np.reshape(obs_interpolation, [num_interpolation, num_interpolation]+self.data.data_shape)
        plot_interpolation(self.opts, obs_interpolation, opts['exp_dir'],
                                    'latent_grid.png',
                                    train=False)

        # - Obs interpolation
        anchors = latents_vizu[anchors_ids].reshape((-1,2,opts['zdim']))
        enc_interpolation = interpolations(anchors,    # shape: [nanchors, nsteps, zdim]
                                    nsteps=num_steps-2,
                                    std=enc_var)
        enc_interpolation = np.reshape(enc_interpolation, [-1,opts['zdim']])
        # reconstructing
        dec_interpolation = self.sess.run(self.generated,
                                    feed_dict={self.pz_samples: enc_interpolation,
                                               self.is_training: False})
        inter_anchors = np.reshape(dec_interpolation, [-1, num_steps-2]+self.data.data_shape)
        obs = self.data.data_vizu[anchors_ids].reshape([-1,2]+self.data.data_shape)
        sobs = obs[:,0].reshape([-1,1]+self.data.data_shape)
        eobs = obs[:,1].reshape([-1,1]+self.data.data_shape)
        inter_anchors = np.concatenate((sobs,inter_anchors,eobs),axis=1)
        plot_interpolation(opts, inter_anchors, opts['exp_dir'],
                                    'interpolations.png',
                                    train=False)

    def perturbation_test(self, WEIGHTS_FILE: typing.Optional[str]=None) -> None:

        opts = self.opts

        # - Load trained model
        if WEIGHTS_FILE is None:
                raise Exception("No model/weights provided")
        else:
            if not tf.gfile.IsDirectory(opts['exp_dir']):
                raise Exception("model doesn't exist")
            WEIGHTS_PATH = os.path.join(opts['exp_dir'],'checkpoints', WEIGHTS_FILE)
            if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_PATH)

        # - Set up
        batches_num = 2
        batch_size = 500
        anchors_ids = [4, 12, 21, 33, 48, 60]
        npert = int(self.data.data_shape[0]/4)
        cost = np.zeros((npert,4))

        if opts['dataset'] == 'shifted_mnist':
            # - Rec/MSE/ground cost vs perturbation
            for _ in range(batches_num):
                # get data and label
                batch_idx = np.random.randint(self.data.test_size,size=batch_size)
                batch_mnist = self.data.all_data[batch_idx]
                batch = np.zeros([batch_size,] + self.data.data_shape)
                pos = np.zeros(batch_size)
                # creating batch data and label
                for n, obs in enumerate(batch_mnist):
                    # padding mnist img
                    paddings = [[2,2], [2,2], [0,0]]
                    obs = np.pad(obs, paddings, mode='constant', constant_values=0.)
                    shape = obs.shape
                    # create img
                    img = np.zeros(self.data.data_shape)
                    # sample cluster pos
                    i = np.random.binomial(1, 0.5)
                    pos_x = i*int(3*shape[0]/8)
                    pos_y = i*int(3*shape[1]/8)
                    # sample shift
                    shift_x = np.random.randint(0, int(shape[0]/8))
                    shift_y = np.random.randint(0, int(shape[1]/8))
                    # place digit
                    img[pos_x+shift_x:shape[0]+pos_x+shift_x, pos_y+shift_y:shape[1]+pos_y+shift_y] = obs
                    batch[n] = img
                    pos[n] = i
                # get shifting direction
                shift_dir = np.stack([2*pos-1,2*pos-1],-1).astype(np.int32)
                for s in range(npert):
                    batch_shifted = shift(opts, batch, shift_dir, 2*s)
                    test_feed_dict={self.inputs_img2: batch,
                                    self.inputs_img1: batch_shifted,
                                    self.is_training: False}
                    c = self.sess.run([self.rec_cost,
                                        self.rec_mse,
                                        self.ground_cost,
                                        self.ground_mse],
                                        feed_dict=test_feed_dict)
                    cost[s] += np.array(c) / batches_num
            rec_cost, mse_cost, ground_cost, ground_mse = np.split(cost,4,-1)
            plot_cost_shift(rec_cost[:,0], mse_cost[:,0], ground_cost[:,0], ground_mse[:,0],
                                    opts['exp_dir'])

            # Plot reconstruction of perturbation
            batch = batch[anchors_ids]
            shift_dir = shift_dir[anchors_ids]
            shifted_obs, shifted_rec, shifted_enc = [], [], []
            for s in range(npert):
                shifted = shift(opts, batch, shift_dir, 2*s)
                [rec,enc] = self.sess.run([self.decoded,self.encoded],
                                        feed_dict={self.inputs_img1: shifted,
                                                   self.is_training: False})
                shifted_obs.append(shifted)
                shifted_rec.append(rec)
                shifted_enc.append(enc)
            shifted_obs = np.stack(shifted_obs,axis=1)
            shifted_rec = np.stack(shifted_rec,axis=1)
            shifted_enc = np.stack(shifted_enc,axis=1)
            plot_rec_shift(opts, shifted_obs, shifted_rec, opts['exp_dir'])
            # plot_embedded_shift(opts, shifted_enc, opts['exp_dir'])
        if opts['dataset'] == 'shifted_3pos_mnist':
            # - Rec/MSE/ground cost vs perturbation
            for _ in range(batches_num):
                # get data and label
                batch_idx = np.random.randint(self.data.test_size,size=batch_size)
                batch_mnist = self.data.all_data[batch_idx]
                batch = np.zeros([batch_size,] + self.data.data_shape)
                pos = np.zeros(batch_size)
                # creating batch data and label
                for n, obs in enumerate(batch_mnist):
                    # padding mnist img
                    paddings = [[2,2], [2,2], [0,0]]
                    obs = np.pad(obs, paddings, mode='constant', constant_values=0.)
                    shape = obs.shape
                    # create img
                    img = np.zeros(self.data.data_shape)
                    # sample cluster pos
                    i = np.random.randint(3)
                    pos_x = i*int(shape[0]/2)
                    pos_y = i*int(shape[1]/2)
                    # place digit
                    img[pos_x:shape[0]+pos_x, pos_y:shape[1]+pos_y] = obs
                    batch[n] = img
                    pos[n] = i
                # get shifting direction
                shift_dir = np.stack([2*pos-1,2*pos-1],-1).astype(np.int32)
                for s in range(npert):
                    batch_shifted = shift(opts, batch, shift_dir, 2*s)
                    test_feed_dict={self.inputs_img2: batch,
                                    self.inputs_img1: batch_shifted,
                                    self.is_training: False}
                    c = self.sess.run([self.rec_cost,
                                        self.rec_mse,
                                        self.ground_cost,
                                        self.ground_mse],
                                        feed_dict=test_feed_dict)
                    cost[s] += np.array(c) / batches_num
            rec_cost, mse_cost, ground_cost, ground_mse = np.split(cost,4,-1)
            plot_cost_shift(rec_cost[:,0], mse_cost[:,0], ground_cost[:,0], ground_mse[:,0],
                                    opts['exp_dir'])

            # Plot reconstruction of perturbation
            batch = batch[anchors_ids]
            shift_dir = shift_dir[anchors_ids]
            shifted_obs, shifted_rec, shifted_enc = [], [], []
            for s in range(npert):
                shifted = shift(opts, batch, shift_dir, 2*s)
                [rec,enc] = self.sess.run([self.decoded,self.encoded],
                                        feed_dict={self.inputs_img1: shifted,
                                                   self.is_training: False})
                shifted_obs.append(shifted)
                shifted_rec.append(rec)
                shifted_enc.append(enc)
            shifted_obs = np.stack(shifted_obs,axis=1)
            shifted_rec = np.stack(shifted_rec,axis=1)
            shifted_enc = np.stack(shifted_enc,axis=1)
            plot_rec_shift(opts, shifted_obs, shifted_rec, opts['exp_dir'])
            # plot_embedded_shift(opts, shifted_enc, opts['exp_dir'])
        elif opts['dataset'] == 'rotated_mnist':
            # - Rec/MSE/ground cost vs perturbation
            for _ in range(batches_num):
                # get data and label
                batch_idx = np.random.randint(self.data.test_size,size=batch_size)
                batch_mnist = self.data.all_data[batch_idx]
                # rotate the data
                # padding mnist img
                paddings = [[0,0], [2,2], [2,2], [0,0]]
                x_pad = np.pad(batch_mnist, paddings, mode='constant', constant_values=0.)
                # rot image with 0.5 prob
                choice = np.random.randint(0,2,batch_size).reshape([batch_size,1,1,1])
                batch = np.where(choice==0, x_pad, np.rot90(x_pad,axes=(1,2)))
                # get rot direction
                rot_dir = (1-2*choice).astype(np.int32).reshape([batch_size,])
                for s in range(npert):
                    batch_rotated = rotate(opts, batch, rot_dir, s, 180./2./npert)
                    test_feed_dict={self.inputs_img2: batch,
                                    self.inputs_img1: batch_rotated,
                                    self.is_training: False}
                    c = self.sess.run([self.rec_cost,
                                        self.rec_mse,
                                        self.ground_cost,
                                        self.ground_mse],
                                        feed_dict=test_feed_dict)
                    cost[s] += np.array(c) / batches_num
            rec_cost, mse_cost, ground_cost, ground_mse = np.split(cost,4,-1)
            plot_cost_shift(rec_cost[:,0], mse_cost[:,0], ground_cost[:,0], ground_mse[:,0],
                                    opts['exp_dir'])

            # Plot reconstruction of perturbation
            batch = batch[anchors_ids]
            rot_dir = rot_dir[anchors_ids]
            rotated_obs, rotated_rec, rotated_enc = [], [], []
            for s in range(npert):
                rotated = rotate(opts, batch, rot_dir, s, 180./2./npert)
                [rec,enc] = self.sess.run([self.decoded,self.encoded],
                                        feed_dict={self.inputs_img1: rotated,
                                                   self.is_training: False})
                rotated_obs.append(rotated)
                rotated_rec.append(rec)
                rotated_enc.append(enc)
            rotated_obs = np.stack(rotated_obs,axis=1)
            rotated_rec = np.stack(rotated_rec,axis=1)
            rotated_enc = np.stack(rotated_enc,axis=1)
            plot_rec_shift(opts, rotated_obs, rotated_rec, opts['exp_dir'])
            # plot_embedded_shift(opts, rotated_enc, opts['exp_dir'])
        elif opts['dataset'] == 'gmm':
            # - Rec/MSE/ground cost vs perturbation
            for _ in range(batches_num):
                    # get data and label
                batch = np.zeros([batch_size,]+self.data.data_shape)
                labels = np.zeros([batch_size,], dtype=int)
                logits_shape = [int(self.data.data_shape[0]/2),int(self.data.data_shape[1]/2),self.data.data_shape[2]]
                for n in range(batch_size):
                    # choose mixture
                    mu = np.zeros(logits_shape)
                    choice = np.random.randint(0,2)
                    mu[3*choice:3*choice+3,3*choice:6*choice+3] = np.ones((3,3,1))
                    mu[1+3*choice,1+3*choice] = [1.5]
                    # sample cat. logits
                    logits = np.random.normal(mu,.1,size=logits_shape).reshape((-1))
                    p = np.exp(logits) / np.sum(np.exp(logits))
                    a = np.arange(np.prod(logits_shape))
                    # sample pixel idx
                    i, j = 0, 0
                    while i==0 or j==0 or i==logits_shape[0]-1 or j==logits_shape[1]-1:
                        idx = np.random.choice(a,size=1,p=p)[0]
                        i = int(idx / 6.)
                        j = idx % 6
                    # generate obs
                    x = np.zeros(datashapes['gmm'])
                    x[2*i:2*i+2,2*i:2*i+2] = np.ones((2,2,1))
                    batch[n] = x
                    labels[n] = choice
                # get shifting direction
                shift_dir = np.stack([2*labels-1,2*labels-1],-1).astype(np.int32)
                for s in range(npert):
                    batch_shifted = shift(opts, batch, shift_dir, s)
                    test_feed_dict={self.inputs_img2: batch,
                                    self.inputs_img1: batch_shifted,
                                    self.is_training: False}
                    c = self.sess.run([self.rec_cost,
                                        self.rec_mse,
                                        self.ground_cost,
                                        self.ground_mse],
                                        feed_dict=test_feed_dict)
                    cost[s] += np.array(c) / batches_num
            rec_cost, mse_cost, ground_cost, ground_mse = np.split(cost,4,-1)
            plot_cost_shift(rec_cost[:,0], mse_cost[:,0], ground_cost[:,0], ground_mse[:,0],
                                    opts['exp_dir'])
            # Plot reconstruction of perturbation
            batch = batch[anchors_ids]
            shift_dir = shift_dir[anchors_ids]
            shifted_obs, shifted_rec, shifted_enc = [], [], []
            for s in range(npert):
                shifted = shift(opts, batch, shift_dir, s)
                [rec,enc] = self.sess.run([self.decoded,self.encoded],
                                        feed_dict={self.inputs_img1: shifted,
                                                   self.is_training: False})
                shifted_obs.append(shifted)
                shifted_rec.append(rec)
                shifted_enc.append(enc)
            shifted_obs = np.stack(shifted_obs,axis=1)
            shifted_rec = np.stack(shifted_rec,axis=1)
            shifted_enc = np.stack(shifted_enc,axis=1)
            plot_rec_shift(opts, shifted_obs, shifted_rec, opts['exp_dir'])
        else:
            assert False, 'Unknown {} dataset'.format(opts['dataset'])

    def fid_score(self, load_trained_model=False, MODEL_PATH=None,
                                        WEIGHTS_FILE=None,
                                        compute_dataset_statistics=False,
                                        fid_inputs='samples',
                                        save_score=False) -> np.ndarray:
        """
        Compute FID score
        """

        opts = self.opts

        # --- Load trained weights
        if load_trained_model:
            if not tf.gfile.IsDirectory(MODEL_PATH):
                raise Exception("model doesn't exist")
            WEIGHTS_PATH = os.path.join(MODEL_PATH,'checkpoints',WEIGHTS_FILE)
            if not tf.gfile.Exists(WEIGHTS_PATH+".meta"):
                raise Exception("weights file doesn't exist")
            self.saver.restore(self.sess, WEIGHTS_PATH)

        if self.data.dataset == 'celebA':
            compare_dataset_name = 'celeba_stats.npz'
        elif self.data.dataset == 'cifar10':
            compare_dataset_name = 'cifar10_stats.npz'
        elif self.data.dataset == 'svhn':
            compare_dataset_name = 'svhn_stats.npz'
        elif self.data.dataset == 'mnist':
            compare_dataset_name = 'mnist_stats.npz'
        else:
            assert False, 'FID not implemented for {} dataset.'.format(self.data.dataset)

        # --- setup
        full_size = self.data.train_size + self.data.test_size
        test_size = 1000
        batch_size = 100
        batch_num = int(test_size/batch_size)
        fid_dir = 'fid'
        stats_path = os.path.join(fid_dir, compare_dataset_name)

        # --- Compute stats on real dataset if needed
        if not os.path.isfile(stats_path) or compute_dataset_statistics:
            preds_list = []
            for n in range(batch_num):
                batch_id = np.random.randint(full_size, size=batch_size)
                batch = self.data._sample_observations(batch_id)
                # rescale inputs in [0,255]
                if self.opts['input_normalize_sym']:
                    batch = batch / 2. + 0.5
                batch *= 255.
                # Convert to RGB if needed
                if np.shape(batch)[-1] == 1:
                    batch = np.repeat(batch, 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': batch})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu = np.mean(preds_list, axis=0)
            sigma = np.cov(preds_list, rowvar=False)
            # saving stats
            np.savez(stats_path, m=mu, s=sigma)
        else:
            stats = np.load(stats_path)
            mu = stats['m']
            sigma = stats['s']

        # --- Compute stats for reconstructions or samples
        if fid_inputs == 'reconstruction':
            preds_list = []
            for n in range(batch_num):
                batch_id = np.random.randint(full_size, size=batch_size)
                batch = self.data._sample_observations(batch_id)
                recons = self.sess.run(self.decoded, feed_dict={
                                        self.inputs_img1: batch,
                                        self.is_training: False})
                # rescale recons in [0,255]
                if self.opts['input_normalize_sym']:
                    recons = recons / 2. + 0.5
                recons *= 255.
                if np.shape(recons)[-1] == 1:
                    recons = np.repeat(recons, 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': recons})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu_model = np.mean(preds_list, axis=0)
            sigma_model = np.cov(preds_list, rowvar=False)
        elif fid_inputs == 'samples':
            preds_list = []
            for n in range(batch_num):
                batch = sample_pz(opts, self.pz_mean, self.pz_Sigma, batch_size)
                samples = self.sess.run(self.generated, feed_dict={
                                        self.pz_samples: batch,
                                        self.is_training: False})
                # rescale samples in [0,255]
                if self.opts['input_normalize_sym']:
                    samples = samples / 2. + 0.5
                samples *= 255.
                if np.shape(samples)[-1] == 1:
                    samples = np.repeat(samples, 3, axis=-1)
                preds_incep = self.inception_sess.run(self.inception_layer,
                              feed_dict={'FID_Inception_Net/ExpandDims:0': samples})
                preds_incep = preds_incep.reshape((batch_size,-1))
                preds_list.append(preds_incep)
            preds_list = np.concatenate(preds_list, axis=0)
            mu_model = np.mean(preds_list, axis=0)
            sigma_model = np.cov(preds_list, rowvar=False)

        # --- Compute FID between real stats and model stats
        fid_scores = calculate_frechet_distance(mu, sigma, mu_model, sigma_model)

        # --- Logging
        debug_str = 'FID={:.3f} for {} data'.format(fid_scores, test_size)
        logging.error(debug_str)

        # --- Saving
        if save_score:
            fid_res_dir = os.path.join(MODEL_PATH,fid_dir)
            if not tf.io.gfile.isdir(fid_res_dir):
                utils.create_dir(fid_res_dir)
            filename = 'fid_' + fid_inputs
            np.save(os.path.join(fid_res_dir,filename),fid_scores)

        return fid_scores
