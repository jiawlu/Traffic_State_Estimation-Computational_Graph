# @author       Jiawei Lu (jiaweil9@asu.edu)
# @time         2021/3/19 13:37
# @desc         [script description]

import tensorflow as tf

class SeqModel(tf.keras.Sequential):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.add(tf.keras.layers.Dense(hidden_sizes[0], activation="relu", input_shape=(input_size,)))
        for hidden_size in hidden_sizes[1:]: self.add(tf.keras.layers.Dense(hidden_size, activation="relu"))
        if output_size is not None: self.add(tf.keras.layers.Dense(output_size))


class Model:
    def __init__(self, shared_hidden_sizes, hidden_sizes_1, hidden_sizes_2, density_scale=10, speed_scale=0.01):
        self.shared_m = SeqModel(2, shared_hidden_sizes, None)
        self.m1 = SeqModel(shared_hidden_sizes[-1], hidden_sizes_1, 1)
        self.m2 = SeqModel(shared_hidden_sizes[-1], hidden_sizes_2, 1)

        # self.trainable_weights = self.shared_m.trainable_weights + self.m1.trainable_weights + self.m2.trainable_weights

        self.vf = tf.Variable(20.0*speed_scale, dtype=tf.float32)
        self.kj = tf.Variable(0.3*density_scale, dtype=tf.float32)

        self.t1mu = tf.Variable(3600.0, dtype=tf.float32)         # t1 of mu
        self.mut1 = tf.Variable(1.0, dtype=tf.float32)            # mu at t1
        self.gammamu = tf.Variable(1e-6, dtype=tf.float32)         # gamma of mu
        self.gamma = tf.Variable(1e-6, dtype=tf.float32)

        # self.trainable_weights = self.shared_m.trainable_weights + self.m1.trainable_weights + self.m2.trainable_weights + \
        #     [self.vf, self.kj] + [self.t1mu, self.mut1, self.gammamu]

        self.trainable_weights = self.shared_m.trainable_weights + self.m1.trainable_weights + self.m2.trainable_weights + \
            [self.vf, self.kj] + [self.t1mu, self.mut1, self.gammamu, self.gamma]

        self.k = None
        self.v = None
        self.q = None
        self.vm = None

        self.mu = None

        self.weights0 = None
        self.weights1 = None
        self.weights2 = None

        self.vf_value = 0.0
        self.kj_value = 0.0

        self.t1mu_value = 0.0         # t1 of mu
        self.mut1_value = 0.0            # mu at t1
        self.gammamu_value = 0.0         # gamma of mu
        self.gamma_value = 0.0


    def __call__(self, x):
        self.k = self.m1(self.shared_m(x))
        self.v = self.m2(self.shared_m(x))
        self.q = self.k * self.v
        self.vm = self.vf * (1.0 - self.k / tf.maximum(0.001, self.kj))



    def saveWeights(self):
        self.weights0 = self.shared_m.get_weights()
        self.weights1 = self.m1.get_weights()
        self.weights2 = self.m2.get_weights()

        self.vf_value = self.vf.numpy()
        self.kj_value = self.kj.numpy()

        self.t1mu_value = self.t1mu.numpy()
        self.mut1_value = self.mut1.numpy()
        self.gammamu_value = self.gammamu.numpy()
        self.gamma_value = self.gamma.numpy()



    def setBestWeights(self):
        self.shared_m.set_weights(self.weights0)
        self.m1.set_weights(self.weights1)
        self.m2.set_weights(self.weights2)

        self.vf = tf.Variable(self.vf_value)
        self.kj = tf.Variable(self.kj_value)

        self.t1mu = tf.Variable(self.t1mu_value)
        self.mut1 = tf.Variable(self.mut1_value)
        self.gammamu = tf.Variable(self.gammamu_value)
        self.gamma = tf.Variable(self.gamma_value)

