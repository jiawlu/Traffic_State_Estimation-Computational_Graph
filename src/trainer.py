# @author       Jiawei Lu (jiaweil9@asu.edu)
# @time         2021/3/19 13:41
# @desc         [script description]


import tensorflow as tf
from datetime import datetime
import random
import time
import numpy as np


class Trainer:
    def __init__(self, data_loader, model):
        self.data_loader = data_loader
        self.variant = data_loader.variant
        self.model = model

        self.t0 = data_loader.t0
        self.t3 = data_loader.t3
        self.max_mu = 2.5
        self.xts = data_loader.xts
        self.fc_coeff = data_loader.fc_coeff

        self.ts_mu = data_loader.ts_mu
        self.mu_coeff = data_loader.mu_coeff

        self.density_scale = self.variant['density_scale']
        self.speed_scale = self.variant['speed_scale']
        self.volume_scale = self.variant['volume_scale']

        self.use_loop = self.variant['use_loop']
        self.use_gps = self.variant['use_gps']
        self.use_bluetooth = self.variant['use_bluetooth']
        self.use_video_ave_density = self.variant['use_video_ave_density']
        self.use_travel_time = self.variant['use_travel_time']

        self.loop_weight = self.variant['loop_weight']
        self.gps_weight = self.variant['gps_weight']
        self.bluetooth_weight = self.variant['bluetooth_weight']
        self.video_ave_density_weight = self.variant['video_ave_density_weight']
        self.travel_time_weight = self.variant['travel_time_weight']
        self.mu_weight = self.variant['mu_weight']
        self.pde_weight = self.variant['pde_weight']
        self.vkm_weight = self.variant['vk_model_weight']
        self.max_q_weight = self.variant['max_q_weight']

        self.number_of_epochs = self.variant['number_of_epochs']
        self.loss_eval_interval = self.variant['loss_eval_interval']
        self.number_of_batches = self.variant['number_of_batches']

        self.opt = tf.optimizers.Adam(learning_rate=self.variant['learning_rate'], amsgrad=self.variant['amsgrad'])
        self.loss_history = []
        self.best_loss = float('inf')
        self.best_loss_dict = {}
        self.best_solution = {}

        now = datetime.now()
        self.date_time = now.strftime("%Y%m%d-%H%M%S")
        self.rnd = random.randint(0,100000)


    @tf.function(input_signature=(tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32)))
    def computeDetectorLoss(self,
                    loop_weights, loop_value,
                    gps_weights, gps_value,
                    bluetooth_weights_k, bluetooth_weights_q,
                    video_ave_density_weights, video_ave_density_value,
                    travel_time_ts, travel_time_value):
        print('tracing')

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.xts)
            self.model(self.xts)

        # loop loss
        q_est = tf.matmul(loop_weights, self.model.q)
        loop_loss = tf.reduce_mean((q_est - loop_value)**2)
        # loop_loss = tf.reduce_mean(tf.abs(q_est - loop_value) / loop_value)

        # gps loss
        v_est_gps = tf.matmul(gps_weights, self.model.v)
        gps_loss = tf.reduce_mean((v_est_gps - gps_value)**2)
        # gps_loss = tf.reduce_mean(tf.abs(v_est_gps - gps_value) / gps_value)

        # bluetooth loss
        count_k = tf.matmul(bluetooth_weights_k, self.model.k)
        count_q = tf.matmul(bluetooth_weights_q, self.model.q)
        bluetooth_loss = tf.reduce_mean((count_k - count_q)**2)

        # video ave density loss
        k_est_vas = tf.matmul(video_ave_density_weights, self.model.k)
        vas_loss = tf.reduce_mean((k_est_vas - video_ave_density_value)**2)
        # vas_loss = tf.reduce_mean(tf.abs(v_est_vas - video_ave_speed_value) / video_ave_speed_value)

        # pde loss
        gradsq = tape.gradient(self.model.q, self.xts) / self.volume_scale
        gradsk = tape.gradient(self.model.k, self.xts) / self.density_scale
        pde_loss = tf.reduce_mean(tf.multiply((gradsq[:,0] + gradsk[:,1])**2, self.fc_coeff))

        # vk model loss
        vkm_loss = tf.reduce_mean((self.model.v - self.model.vm)**2)

        # mu loss
        macro_mu = self.model.mut1 + self.model.gammamu*(self.ts_mu-self.model.t1mu)**2
        meso_mu = tf.matmul(self.mu_coeff, self.model.q)
        mu_loss = tf.reduce_mean((macro_mu * self.volume_scale - meso_mu)**2)

        # travel time loss
        fftt = 5000 / (self.model.vf / self.speed_scale)        # second
        virtual_arrival_time = travel_time_ts + fftt

        w1 = self.Qt(virtual_arrival_time) / self.max_mu
        w2 = (self.Qt(virtual_arrival_time) - self.Dt(virtual_arrival_time + w1) + self.Dt(virtual_arrival_time)) / self.max_mu + w1
        w3 = (self.Qt(virtual_arrival_time) - self.Dt(virtual_arrival_time + w2) + self.Dt(virtual_arrival_time)) / self.max_mu + w2
        w4 = (self.Qt(virtual_arrival_time) - self.Dt(virtual_arrival_time + w3) + self.Dt(virtual_arrival_time)) / self.max_mu + w3
        w5 = (self.Qt(virtual_arrival_time) - self.Dt(virtual_arrival_time + w4) + self.Dt(virtual_arrival_time)) / self.max_mu + w4
        wt = (self.Qt(virtual_arrival_time) - self.Dt(virtual_arrival_time + w5) + self.Dt(virtual_arrival_time)) / self.max_mu + w5
        tt = fftt + wt
        travel_time_loss = tf.reduce_mean((tt - travel_time_value) ** 2)

        # travel_time_loss = tf.constant(0.0, dtype=tf.float32)

        # max q loss
        max_q_loss = tf.reduce_mean(tf.nn.relu(macro_mu - self.max_mu)**2)


        return loop_loss, gps_loss, bluetooth_loss, vas_loss, pde_loss, vkm_loss, mu_loss, travel_time_loss, max_q_loss

    def Qt(self, t):
        t_proj = tf.minimum(self.t3, tf.maximum(self.t0,t))
        return self.model.gamma/3*(t_proj-self.t0)**2*(self.t3-t_proj)

    def Dt(self, t):
        return self.model.mut1*t+self.model.gammamu/3*(t-self.model.t1mu)**3 - (self.model.mut1*self.t0+self.model.gammamu/3*(self.t0-self.model.t1mu)**3)

    def computeLoss(self, loop_weights, loop_value,
                    gps_weights, gps_value,
                    bluetooth_weights_k, bluetooth_weights_q,
                    video_ave_density_weights, video_ave_density_value,
                    travel_time_ts, travel_time_value):
        loop_loss, gps_loss, bluetooth_loss, vas_loss, pde_loss, vkm_loss, mu_loss, travel_time_loss, max_q_loss = self.computeDetectorLoss(loop_weights, loop_value,
            gps_weights, gps_value, bluetooth_weights_k, bluetooth_weights_q, video_ave_density_weights, video_ave_density_value, travel_time_ts, travel_time_value)

        total_loss = self.loop_weight * loop_loss + self.gps_weight * gps_loss + self.bluetooth_weight * bluetooth_loss + \
            self.video_ave_density_weight * vas_loss + self.pde_weight * pde_loss + self.vkm_weight * vkm_loss + self.mu_weight * mu_loss + \
            self.travel_time_weight * travel_time_loss + self.max_q_weight * max_q_loss
        return total_loss


    @tf.function(input_signature=(tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None,None], dtype=tf.float32),tf.TensorSpec(shape=[None,None], dtype=tf.float32)))
    def step(self, loop_weights, loop_value, gps_weights, gps_value, bluetooth_weights_k, bluetooth_weights_q,
             video_ave_density_weights, video_ave_density_value, travel_time_ts, travel_time_value):
        with tf.GradientTape() as tape:
            loss = self.computeLoss(loop_weights, loop_value, gps_weights, gps_value, bluetooth_weights_k, bluetooth_weights_q,
                                    video_ave_density_weights, video_ave_density_value, travel_time_ts, travel_time_value)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.opt.apply_gradients(zip(grads, self.model.trainable_weights))


    def saveLossAndSolution(self, epoch):
        loop_loss, gps_loss, bluetooth_loss, vas_loss, pde_loss, vkm_loss, mu_loss, travel_time_loss, max_q_loss = \
            self.computeDetectorLoss(self.data_loader.loop_weights, self.data_loader.loop_value,
                                     self.data_loader.gps_weights, self.data_loader.gps_value,
                                     self.data_loader.bluetooth_weights_k, self.data_loader.bluetooth_weights_q,
                                     self.data_loader.video_ave_density_weights, self.data_loader.video_ave_density_value,
                                     self.data_loader.travel_time_ts, self.data_loader.travel_time_value)

        loop_loss_value = loop_loss.numpy()
        gps_loss_value = gps_loss.numpy()
        bluetooth_loss_value = bluetooth_loss.numpy()
        vas_loss_value = vas_loss.numpy()
        pde_loss_value = pde_loss.numpy()
        vkm_loss_value = vkm_loss.numpy()
        mu_loss_value = mu_loss.numpy()
        travel_time_loss_value = travel_time_loss.numpy()
        max_q_loss_value = max_q_loss.numpy()
        loss_value = self.loop_weight * loop_loss_value + self.gps_weight * gps_loss_value + self.bluetooth_weight * bluetooth_loss_value + \
            self.video_ave_density_weight * vas_loss_value + self.pde_weight * pde_loss_value + self.vkm_weight * vkm_loss_value + \
            self.mu_weight * mu_loss_value + self.travel_time_weight * travel_time_loss_value + self.max_q_weight * max_q_loss_value

        loss_dict = {'epoch': epoch, 'total_loss': loss_value, 'loop': loop_loss_value, 'gps': gps_loss_value,
                     'bluetooth': bluetooth_loss_value, 'vas': vas_loss_value, 'pde': pde_loss_value,
                     'vkm': vkm_loss_value, 'mu':mu_loss_value, 'travel time':travel_time_loss_value, 'maxq':max_q_loss_value}

        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_loss_dict = loss_dict
            self.model.saveWeights()

        loss_dict['best_loss'] = self.best_loss
        self.loss_history.append(loss_dict)

        print('epoch {:<8}| best:{:<10.3g}| loss:{:<10.3g} loop:{:<10.3g} gps:{:<10.3g} bt:{:<10.3g} vas:{:<10.3g} pde:{:<10.3g} vkm:{:<10.3g} mu:{:<10.3g} travel time:{:<10.3g} maxq:{:<10.3g}'.format(
            epoch, self.best_loss, loss_value, loop_loss_value, gps_loss_value, bluetooth_loss_value, vas_loss_value, pde_loss_value, vkm_loss_value, mu_loss_value, travel_time_loss_value, max_q_loss_value))


    def train(self):
        t_start = time.time()

        self.saveLossAndSolution(0)

        for epoch in range(1, self.number_of_epochs + 1):
            # print(f'  epoch: {epoch}/{self.number_of_epochs}')

            for batch_no in range(self.number_of_batches):
                # print(f'  batch: {batch_no+1}/{self.number_of_batches}')

                loop_weights, loop_value = self.data_loader.loop_batch_list[batch_no]
                gps_weights, gps_value = self.data_loader.gps_batch_list[batch_no]
                bluetooth_weights_k, bluetooth_weights_q = self.data_loader.bluetooth_batch_list[batch_no]
                video_ave_density_weights, video_ave_density_value = self.data_loader.video_batch_list[batch_no]
                travel_time_ts, travel_time_value = self.data_loader.travel_time_batch_list[batch_no]

                gamma_value = self.model.gamma.numpy()
                gammamu_value = self.model.gammamu.numpy()
                self.step(loop_weights, loop_value, gps_weights, gps_value, bluetooth_weights_k, bluetooth_weights_q,
                          video_ave_density_weights, video_ave_density_value, travel_time_ts, travel_time_value)
                if self.model.gamma < gamma_value - 1e-8:
                    self.model.gamma.assign(gamma_value - 1e-8)
                elif self.model.gamma > gamma_value + 1e-8:
                    self.model.gamma.assign(gamma_value + 1e-8)
                if self.model.gammamu < gammamu_value - 1e-8:
                    self.model.gammamu.assign(gammamu_value - 1e-8)
                elif self.model.gammamu > gammamu_value + 1e-8:
                    self.model.gammamu.assign(gammamu_value + 1e-8)

                print(f'gamma:{self.model.gamma.numpy()}, gammamu:{self.model.gammamu.numpy()}, mut1:{self.model.mut1.numpy()}, t1mu:{self.model.t1mu.numpy()}, vf:{self.model.vf.numpy()}')

            if epoch % self.loss_eval_interval == 0:
                self.saveLossAndSolution(epoch)

        t_end = time.time()
        print('total time: {}'.format(t_end - t_start))

        self.model.setBestWeights()
        self.model(self.xts)
