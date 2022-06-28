# @author       Jiawei Lu (jiaweil9@asu.edu)
# @time         2021/3/19 12:08
# @desc         [script description]

import tensorflow as tf
import numpy as np
import os
import csv
import math


class DataLoader:
    def __init__(self, variant):
        self.variant = variant

        self.number_of_batches = variant['number_of_batches']

        self.loop_weights = None
        self.loop_value = None
        self.gps_weights = None
        self.gps_value = None
        self.bluetooth_weights_k = None
        self.bluetooth_weights_q = None
        self.video_ave_density_weights = None
        self.video_ave_density_value = None
        self.travel_time_ts = None
        self.travel_time_value = None

        self.loop_batch_list = []
        self.gps_batch_list = []
        self.bluetooth_batch_list = []
        self.video_batch_list = []
        self.fc_coeff = None
        self.travel_time_batch_list = []

        self.number_of_loop_data_records = 1
        self.number_of_gps_data_records = 1
        self.number_of_bluetooth_data_records = 1
        self.number_of_video_ave_density_data_records = 1
        self.number_of_travel_time_records = 1

        self.number_of_x_points = int((variant['space_end'] - variant['space_start']) / variant['grid_space_length']) + 1
        self.number_of_t_points = int((variant['time_end'] - variant['time_start']) / variant['grid_time_length']) + 1
        self.number_of_xt_points = self.number_of_x_points * self.number_of_t_points

        self.xs = np.linspace(variant['space_start'], variant['space_end'], self.number_of_x_points, endpoint=True)
        self.ts = np.linspace(variant['time_start'], variant['time_end'], self.number_of_t_points, endpoint=True)
        self.xts_array = np.array([(x, t) for x in self.xs for t in self.ts])
        self.xts = tf.constant(self.xts_array * np.array([variant['x_scale'], variant['t_scale']]), dtype=tf.float32)

        self.t0 = 2100.0
        self.t3 = 6180.0
        t_start_idx = math.ceil(self.t0 / variant['grid_time_length'])
        t_end_idx = math.floor(self.t3 / variant['grid_time_length'])
        number_of_ts_mu = t_end_idx - t_start_idx + 1
        self.ts_mu = tf.constant(np.array([t_idx*variant['grid_time_length'] for t_idx in range(t_start_idx, t_end_idx+1)]).reshape(-1, 1), tf.float32)
        mu_coeff = np.zeros((number_of_ts_mu, self.number_of_xt_points), dtype=np.float32)
        mu_coeff[:,self.number_of_t_points*(self.number_of_x_points-1)+t_start_idx:self.number_of_t_points*(self.number_of_x_points-1)+t_end_idx+1] = np.identity(number_of_ts_mu)
        self.mu_coeff = tf.constant(mu_coeff, tf.float32)

        self.loadDataset()
        self.batchDataset()


    def getBoxBoundaryIndex(self, x, t):
        x_end_idx = np.where(self.xs >= x)[0][0]
        x_start_idx = x_end_idx - 1
        t_end_idx = np.where(self.ts >= t)[0][0]
        t_start_idx = t_end_idx - 1

        dl_idx = x_start_idx * self.number_of_t_points + t_start_idx
        dr_idx = x_start_idx * self.number_of_t_points + t_end_idx
        ul_idx = x_end_idx * self.number_of_t_points + t_start_idx
        ur_idx = x_end_idx * self.number_of_t_points + t_end_idx

        x_down_ratio = (self.xs[x_end_idx] - x) / self.variant['grid_space_length']
        x_up_ratio = (x - self.xs[x_start_idx]) / self.variant['grid_space_length']
        t_left_ratio = (self.ts[t_end_idx] - t) / self.variant['grid_time_length']
        t_right_ratio = (t - self.ts[t_start_idx]) / self.variant['grid_time_length']

        dl_weight = x_down_ratio * t_left_ratio
        dr_weight = x_down_ratio * t_right_ratio
        ul_weight = x_up_ratio * t_left_ratio
        ur_weight = x_up_ratio * t_right_ratio

        xs_idxs = [dl_idx, dr_idx, ul_idx, ur_idx]
        weights = [dl_weight, dr_weight, ul_weight, ur_weight]
        return xs_idxs, weights

    def getSegmentCount(self, t, x1, x2):
        xs_idxs_1, weights_1 = self.getBoxBoundaryIndex(x1, t)

        t_idx_right = np.where(self.ts >= t)[0][0]
        t_idx_left = t_idx_right - 1
        t_left_ratio = (self.ts[t_idx_right] - t) / self.variant['grid_time_length']
        t_right_ratio = (t - self.ts[t_idx_left]) / self.variant['grid_time_length']

        inbetween_x_idx_start = np.where(self.xs > x1)[0][0]
        inbetween_x_idx_end = np.where(self.xs < x2)[0][-1]

        xs_idxs_m_list = []
        weights_m_list = []
        for x_idx in range(inbetween_x_idx_start, inbetween_x_idx_end + 1):
            l_idx = x_idx * self.number_of_t_points + t_idx_left
            r_idx = x_idx * self.number_of_t_points + t_idx_right
            xs_idxs_m_list.append([l_idx, r_idx])
            weights_m_list.append([t_left_ratio, t_right_ratio])

        xs_idxs_2, weights_2 = self.getBoxBoundaryIndex(x2, t)

        d1 = self.xs[inbetween_x_idx_start] - x1
        d2 = x2 - self.xs[inbetween_x_idx_end]

        xs_idxs = xs_idxs_1.copy()
        for xs_idxs_m in xs_idxs_m_list: xs_idxs += xs_idxs_m
        xs_idxs += xs_idxs_2

        weights = [weight / 2 * d1 for weight in weights_1]
        weights += [weight / 2 * (d1 + self.variant['grid_space_length']) for weight in weights_m_list[0]]
        for weight_m in weights_m_list[1:-1]: weights += [weight * self.variant['grid_space_length'] for weight in weight_m]
        weights += [weight / 2 * (d2 + self.variant['grid_space_length']) for weight in weights_m_list[-1]]
        weights += [weight / 2 * d2 for weight in weights_2]

        weights = [weight / self.variant['density_scale'] for weight in weights]
        return xs_idxs, weights

    def getSectionFlow(self, x, t1, t2):
        xs_idxs_1, weights_1 = self.getBoxBoundaryIndex(x, t1)

        x_idx_up = np.where(self.xs >= x)[0][0]
        x_idx_down = x_idx_up - 1
        x_down_ratio = (self.xs[x_idx_up] - x) / self.variant['grid_space_length']
        x_up_ratio = (x - self.xs[x_idx_down]) / self.variant['grid_space_length']

        inbetween_t_idx_start = np.where(self.ts > t1)[0][0]
        inbetween_t_idx_end = np.where(self.ts < t2)[0][-1]

        xs_idxs_m_list = []
        weights_m_list = []
        for t_idx in range(inbetween_t_idx_start, inbetween_t_idx_end + 1):
            d_idx = x_idx_down * self.number_of_t_points + t_idx
            u_idx = x_idx_up * self.number_of_t_points + t_idx
            xs_idxs_m_list.append([d_idx, u_idx])
            weights_m_list.append([x_down_ratio, x_up_ratio])

        xs_idxs_2, weights_2 = self.getBoxBoundaryIndex(x, t2)

        d1 = self.ts[inbetween_t_idx_start] - t1
        d2 = t2 - self.ts[inbetween_t_idx_end]

        xs_idxs = xs_idxs_1.copy()
        for xs_idxs_m in xs_idxs_m_list: xs_idxs += xs_idxs_m
        xs_idxs += xs_idxs_2

        weights = [weight / 2 * d1 for weight in weights_1]
        weights += [weight / 2 * (d1 + self.variant['grid_time_length']) for weight in weights_m_list[0]]
        for weight_m in weights_m_list[1:-1]:
            weights += [weight * self.variant['grid_time_length'] for weight in weight_m]
        weights += [weight / 2 * (d2 + self.variant['grid_time_length']) for weight in weights_m_list[-1]]
        weights += [weight / 2 * d2 for weight in weights_2]

        weights = [weight / self.variant['volume_scale'] for weight in weights]
        return xs_idxs, weights


    def loadDataset(self):
        if self.variant['use_loop']:
            loop_weights_list = []
            loop_value_list = []
            loop_filepath = os.path.join(self.variant['data_directory'], 'loop.csv')
            with open(loop_filepath) as fin:
                reader = csv.DictReader(fin)
                for line in reader:
                    location = float(line['location'])
                    t_start = float(line['t_start'])
                    t_end = float(line['t_end'])
                    count = float(line['count'])
                    volume = count / (t_end - t_start) * self.variant['volume_scale']

                    x_idx = round((location - self.variant['space_start']) / self.variant['grid_space_length'])
                    t_start_idx = round((t_start - self.variant['time_start']) / self.variant['grid_time_length'])
                    t_end_idx = round((t_end - self.variant['time_start']) / self.variant['grid_time_length'])
                    xs_start_idx = x_idx * self.number_of_t_points + t_start_idx
                    xs_end_idx = x_idx * self.number_of_t_points + t_end_idx
                    xs_idxs = list(range(xs_start_idx, xs_end_idx + 1))

                    nb_xs_points = len(xs_idxs)
                    weights = [0.5 / (nb_xs_points - 1)] + [1 / (nb_xs_points - 1)] * (nb_xs_points - 2) + [0.5 / (nb_xs_points - 1)]

                    weight_array = np.zeros((1, self.number_of_xt_points), dtype=np.float32)
                    weight_array[:,xs_idxs] = weights
                    loop_weights_list.append(weight_array)
                    loop_value_list.append(volume)

            self.number_of_loop_data_records = len(loop_weights_list)
            self.loop_weights = tf.constant(np.concatenate(loop_weights_list), tf.float32)
            self.loop_value = tf.constant(np.array(loop_value_list).reshape(-1,1), tf.float32)
        else:
            self.loop_weights = tf.zeros((1, self.number_of_xt_points), tf.float32)
            self.loop_value = tf.zeros((1, 1), tf.float32)


        if self.variant['use_gps']:
            gps_weights_list = []
            gps_value_list = []
            gps_filepath = os.path.join(self.variant['data_directory'], 'gps.csv')
            with open(gps_filepath) as fin:
                reader = csv.DictReader(fin)
                for line in reader:
                    t = float(line['time'])
                    x = float(line['pos'])
                    v = float(line['speed']) * self.variant['speed_scale']
                    xs_idxs, weights = self.getBoxBoundaryIndex(x, t)

                    weight_array = np.zeros((1, self.number_of_xt_points), dtype=np.float32)
                    weight_array[:,xs_idxs] = weights
                    gps_weights_list.append(weight_array)
                    gps_value_list.append(v)

            self.number_of_gps_data_records = len(gps_weights_list)
            self.gps_weights = tf.constant(np.concatenate(gps_weights_list), tf.float32)
            self.gps_value = tf.constant(np.array(gps_value_list).reshape(-1,1), tf.float32)
        else:
            self.gps_weights = tf.zeros((1, self.number_of_xt_points), tf.float32)
            self.gps_value = tf.zeros((1, 1), tf.float32)


        if self.variant['use_bluetooth']:
            bt_weights_k_list = []
            bt_weights_q_list = []
            bluetooth_filepath = os.path.join(self.variant['data_directory'], 'bluetooth.csv')
            bluetooth_data_dict = {}
            with open(bluetooth_filepath) as fin:
                reader = csv.DictReader(fin)
                for line in reader:
                    veh_id = line['veh_id']
                    t = float(line['time'])
                    x = float(line['x'])
                    if veh_id in bluetooth_data_dict.keys():
                        bluetooth_data_dict[veh_id].append({'t': t, 'x': x})
                    else:
                        bluetooth_data_dict[veh_id] = [{'t': t, 'x': x}]

            for veh_id, bt_veh in bluetooth_data_dict.items():
                if len(bt_veh) < 2: continue
                for i in range(len(bt_veh) - 1):
                    t1, x1 = bt_veh[i]['t'], bt_veh[i]['x']
                    t2, x2 = bt_veh[i + 1]['t'], bt_veh[i + 1]['x']
                    xs_idxs_k, weights_k = self.getSegmentCount(t1, x1, x2)
                    xs_idxs_q, weights_q = self.getSectionFlow(x2, t1, t2)

                    weight_array_k = np.zeros((1, self.number_of_xt_points), dtype=np.float32)
                    for idx_no, xs_idx in enumerate(xs_idxs_k): weight_array_k[0,xs_idx] += weights_k[idx_no]
                    bt_weights_k_list.append(weight_array_k)
                    weight_array_q = np.zeros((1, self.number_of_xt_points), dtype=np.float32)
                    for idx_no, xs_idx in enumerate(xs_idxs_q): weight_array_q[0, xs_idx] += weights_q[idx_no]
                    bt_weights_q_list.append(weight_array_q)

            self.number_of_bluetooth_data_records = len(bt_weights_k_list)
            self.bluetooth_weights_k = tf.constant(np.concatenate(bt_weights_k_list), tf.float32)
            self.bluetooth_weights_q = tf.constant(np.concatenate(bt_weights_q_list), tf.float32)
        else:
            self.bluetooth_weights_k = tf.zeros((1, self.number_of_xt_points), tf.float32)
            self.bluetooth_weights_q = tf.zeros((1, self.number_of_xt_points), tf.float32)


        if self.variant['use_video_ave_density']:
            vas_weights_list = []
            vas_value_list = []
            video_ave_density_filepath = os.path.join(self.variant['data_directory'], 'video_ave_density.csv')
            with open(video_ave_density_filepath) as fin:
                reader = csv.DictReader(fin)
                for line in reader:
                    x1 = float(line['x_start'])
                    x2 = float(line['x_end'])
                    t1 = float(line['t_start'])
                    t2 = float(line['t_end'])
                    k = float(line['density']) * self.variant['density_scale']

                    x1_idx = int((x1 - self.variant['space_start']) / self.variant['grid_space_length'])
                    x2_idx = int((x2 - self.variant['space_start']) / self.variant['grid_space_length'])
                    t1_idx = int((t1 - self.variant['time_start']) / self.variant['grid_time_length'])
                    t2_idx = int((t2 - self.variant['time_start']) / self.variant['grid_time_length'])

                    dl_idx = x1_idx * self.number_of_t_points + t1_idx
                    dr_idx = x1_idx * self.number_of_t_points + t2_idx
                    ul_idx = x2_idx * self.number_of_t_points + t1_idx
                    ur_idx = x2_idx * self.number_of_t_points + t2_idx
                    xs_idxs = [dl_idx, dr_idx, ul_idx, ur_idx]
                    weights = [0.25] * 4

                    weight_array = np.zeros((1, self.number_of_xt_points), dtype=np.float32)
                    weight_array[:,xs_idxs] = weights
                    vas_weights_list.append(weight_array)
                    vas_value_list.append(k)

            self.number_of_video_ave_density_data_records = len(vas_weights_list)
            self.video_ave_density_weights = tf.constant(np.concatenate(vas_weights_list), tf.float32)
            self.video_ave_density_value = tf.constant(np.array(vas_value_list).reshape(-1,1), tf.float32)
        else:
            self.video_ave_density_weights = tf.zeros((1, self.number_of_xt_points), tf.float32)
            self.video_ave_density_value = tf.zeros((1, 1), tf.float32)


        if self.variant['use_travel_time']:
            ts_list = []
            tt_list = []
            travel_time_filepath = os.path.join(self.variant['data_directory'], 'travel_time.csv')
            with open(travel_time_filepath) as fin:
                reader = csv.DictReader(fin)
                for line in reader:
                    t = float(line['time'])
                    travel_time = float(line['travel_time'])

                    ts_list.append(t)
                    tt_list.append(travel_time)

            self.number_of_travel_time_records = len(ts_list)
            self.travel_time_ts = tf.constant(np.array(ts_list).reshape(-1,1), tf.float32)
            self.travel_time_value = tf.constant(np.array(tt_list).reshape(-1, 1), tf.float32)
        else:
            pass

        fc_coeff_array = np.ones([self.number_of_xt_points, 1], dtype=np.float32)
        ramp_filepath = os.path.join(self.variant['data_directory'], 'ramp.csv')
        with open(ramp_filepath) as fin:
            reader = csv.DictReader(fin)
            for line in reader:
                locations_str = line['locations']
                locations = list(map(float,locations_str.split(';')))
                for location in locations:
                    x_idx = (location - self.variant['space_start']) / self.variant['grid_space_length']
                    x_idx1, x_idx2 = int(np.floor(x_idx)), int(np.ceil(x_idx))
                    fc_coeff_array[x_idx1*self.number_of_t_points:(x_idx1+1)*self.number_of_t_points] = 0.0
                    fc_coeff_array[x_idx2*self.number_of_t_points:(x_idx2+1)*self.number_of_t_points] = 0.0
        self.fc_coeff = tf.constant(fc_coeff_array, tf.float32)


    def batchDataset(self):
        self.loop_batch_list = getBatchList(self.number_of_loop_data_records, self.number_of_batches, (self.loop_weights, self.loop_value))
        self.gps_batch_list = getBatchList(self.number_of_gps_data_records, self.number_of_batches, (self.gps_weights, self.gps_value))
        self.bluetooth_batch_list = getBatchList(self.number_of_bluetooth_data_records, self.number_of_batches, (self.bluetooth_weights_k, self.bluetooth_weights_q))
        self.video_batch_list = getBatchList(self.number_of_video_ave_density_data_records, self.number_of_batches, (self.video_ave_density_weights, self.video_ave_density_value))
        self.travel_time_batch_list = getBatchList(self.number_of_travel_time_records, self.number_of_batches, (self.travel_time_ts, self.travel_time_value))


def getBatchList(number_of_records, number_of_batches, record_matrix_list):
    batch_size = math.ceil(number_of_records/number_of_batches)

    batch_list = []
    end_idx = -1
    for i in range(number_of_batches):
        start_idx = min(end_idx+1, number_of_records-1)
        end_idx = min(start_idx+batch_size-1, number_of_records-1)
        aa = []
        for record_matrix in record_matrix_list:
            aa.append(record_matrix[start_idx:end_idx+1,:])
        batch_list.append(aa)
    return batch_list



