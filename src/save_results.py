# @author       Jiawei Lu (jiaweil9@asu.edu)
# @time         2021/3/19 14:00
# @desc         [script description]

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def outputResults(trainer):
    variant = trainer.variant
    data_loader = trainer.data_loader

    output_directory = variant['output_directory']
    if not os.path.exists(output_directory): os.mkdir(output_directory)
    output_identifier = f'output_{trainer.date_time}_{trainer.rnd}'

    # loss
    loss_filepath = os.path.join(output_directory, f'{output_identifier}_loss.csv')
    loss_df = pd.DataFrame(trainer.loss_history)
    loss_df.to_csv(loss_filepath, index=False)

    # raw estimation
    density_scale, speed_scale, volume_scale = variant['density_scale'], variant['speed_scale'], variant['volume_scale']
    k_est = trainer.model.k.numpy() / density_scale
    v_est = trainer.model.v.numpy() / speed_scale
    q_est = trainer.model.q.numpy() / volume_scale

    # comparison with the observation
    if variant['use_loop']:
        detector_value = data_loader.loop_value.numpy() / volume_scale
        volume_est = np.matmul(data_loader.loop_weights.numpy(), q_est)
        loop_comp_df = pd.DataFrame(np.concatenate([detector_value, volume_est], axis=1), columns=['observation', 'estimation'])
        loop_comp_filepath = os.path.join(output_directory, f'{output_identifier}_loop.csv')
        loop_comp_df.to_csv(loop_comp_filepath, index=False)
        loop_mape = (abs(loop_comp_df['estimation'] - loop_comp_df['observation']) / loop_comp_df['observation']).mean()
        loop_mape = round(loop_mape * 100, 2)

        loop_weight_filepath = os.path.join(output_directory, f'{output_identifier}_loop weight.csv')
        np.savetxt(loop_weight_filepath, data_loader.loop_weights.numpy(), fmt='%.3f', delimiter=',')
    else:
        loop_mape = None


    if variant['use_gps']:
        detector_value = data_loader.gps_value.numpy() / speed_scale
        gps_est = np.matmul(data_loader.gps_weights.numpy(), v_est)
        gps_comp_df = pd.DataFrame(np.concatenate([detector_value, gps_est], axis=1), columns=['observation', 'estimation'])
        gps_comp_filepath = os.path.join(output_directory, f'{output_identifier}_gps.csv')
        gps_comp_df.to_csv(gps_comp_filepath, index=False)
        gps_mape = (abs(gps_comp_df['estimation'] - gps_comp_df['observation']) / gps_comp_df['observation']).mean()
        gps_mape = round(gps_mape * 100, 2)
    else:
        gps_mape = None

    if variant['use_bluetooth']:
        bt_k = np.matmul(data_loader.bluetooth_weights_k.numpy()*density_scale, k_est)
        bt_q = np.matmul(data_loader.bluetooth_weights_q.numpy()*volume_scale, q_est)
        bt_comp_df = pd.DataFrame(np.concatenate([bt_k, bt_q], axis=1), columns=['count_k', 'count_q'])
        bt_comp_filepath = os.path.join(output_directory, f'{output_identifier}_bluetooth.csv')
        bt_comp_df.to_csv(bt_comp_filepath, index=False)
        bt_mape = (abs(bt_comp_df['count_k'] - bt_comp_df['count_q']) / bt_comp_df['count_k']).mean()
        bt_mape = round(bt_mape * 100, 2)
    else:
        bt_mape = None

    if variant['use_video_ave_density']:
        detector_value = data_loader.video_ave_density_value.numpy() / density_scale
        vas_est = np.matmul(data_loader.video_ave_density_weights.numpy(), k_est)
        vas_comp_df = pd.DataFrame(np.concatenate([detector_value, vas_est], axis=1), columns=['observation', 'estimation'])
        vas_comp_filepath = os.path.join(output_directory, f'{output_identifier}_video_ave_density.csv')
        vas_comp_df.to_csv(vas_comp_filepath, index=False)
        vas_mape = (abs(vas_comp_df['estimation'] - vas_comp_df['observation']) / vas_comp_df['observation']).mean()
        vas_mape = round(vas_mape * 100, 2)
    else:
        vas_mape = None

    macro_mu = trainer.model.mut1_value + trainer.model.gammamu_value * (data_loader.ts_mu.numpy() - trainer.model.t1mu_value) ** 2
    meso_mu = np.matmul(data_loader.mu_coeff.numpy(), q_est)
    mu_comp_df = pd.DataFrame(np.concatenate([macro_mu, meso_mu], axis=1), columns=['macro_mu', 'meso_mu'])
    mu_comp_filepath = os.path.join(output_directory, f'{output_identifier}_mu.csv')
    mu_comp_df.to_csv(mu_comp_filepath, index=False)

    # settings and loss
    settings_filepath = os.path.join(output_directory, f'{output_identifier}_settings and loss.txt')
    with open(settings_filepath, 'w') as fin:
        for setting_key, setting_value in variant.items():
            fin.write(f'{setting_key}: {setting_value}\n')
        fin.write('\n')

        fin.write(f'free flow speed: {trainer.model.vf.numpy() / speed_scale}\n')
        fin.write(f'jam density: {trainer.model.kj.numpy() / density_scale}\n')
        fin.write(f't0: {trainer.t0}\n')
        fin.write(f't3: {trainer.t3}\n')
        fin.write(f't1 of mu: {trainer.model.t1mu.numpy()}\n')
        fin.write(f'mu at t1: {trainer.model.mut1.numpy()}\n')
        fin.write(f'gamma of mu: {trainer.model.gammamu.numpy()}\n')
        fin.write(f'gamma: {trainer.model.gamma.numpy()}\n')
        fin.write('\n')

        for setting_key, setting_value in trainer.best_loss_dict.items():
            fin.write(f'{setting_key}: {setting_value}\n')
        fin.write('\n')

        fin.write(f'loop_mape: {loop_mape}\n')
        fin.write(f'gps_mape: {gps_mape}\n')
        fin.write(f'bluetooth_mape: {bt_mape}\n')
        fin.write(f'video_ave_density_mape: {vas_mape}\n')


    # density and speed
    xts_array = data_loader.xts_array
    qkv = np.concatenate([xts_array, q_est, k_est, v_est], axis=1)
    qkv_df = pd.DataFrame(qkv, columns=['x','t','volume','density','speed'])
    qkv_filepath = os.path.join(output_directory, f'{output_identifier}_qkv_points.csv')
    qkv_df.to_csv(qkv_filepath, index=False)

    k_mat = k_est.reshape(data_loader.number_of_x_points, -1)
    k_profile = np.zeros((data_loader.number_of_x_points - 1, data_loader.number_of_t_points - 1))
    for i in range(data_loader.number_of_x_points - 1):
        for j in range(data_loader.number_of_t_points - 1):
            k_profile[i, j] = (k_mat[i, j] + k_mat[i, j + 1] + k_mat[i + 1, j] + k_mat[i + 1, j + 1]) / 4

    v_mat = v_est.reshape(data_loader.number_of_x_points, -1)
    v_profile = np.zeros((data_loader.number_of_x_points - 1, data_loader.number_of_t_points - 1))
    for i in range(data_loader.number_of_x_points - 1):
        for j in range(data_loader.number_of_t_points - 1):
            v_profile[i, j] = (v_mat[i, j] + v_mat[i, j + 1] + v_mat[i + 1, j] + v_mat[i + 1, j + 1]) / 4

    v_profile = np.flipud(v_profile)
    v_profile_filepath = os.path.join(output_directory, f'{output_identifier}_vprofile.csv')
    np.savetxt(v_profile_filepath, v_profile, fmt='%.2f', delimiter=',')

    plt.figure(figsize=(25, 10))
    sns.heatmap(data=v_profile, vmin=0, vmax=20, cmap='RdYlGn')
    picpath = os.path.join(output_directory, f'{output_identifier}_vprofile.jpg')
    plt.savefig(picpath)
    plt.close()

    k_profile = np.flipud(k_profile)
    k_profile_filepath = os.path.join(output_directory, f'{output_identifier}_kprofile.csv')
    np.savetxt(k_profile_filepath, k_profile, fmt='%.4f', delimiter=',')

    plt.figure(figsize=(25, 10))
    sns.heatmap(data=k_profile, vmin=0, vmax=0.5, cmap='RdYlGn')
    picpath = os.path.join(output_directory, f'{output_identifier}_kprofile.jpg')
    plt.savefig(picpath)
    plt.close()


    k_est_cell, v_est_cell = k_profile.reshape(-1,1), v_profile.reshape(-1,1)
    delta_x, delta_t = data_loader.variant['grid_space_length'], data_loader.variant['grid_time_length']
    xts_array_cell = np.array([(x, x+delta_x, t, t+delta_t) for x in data_loader.xs[:-1] for t in data_loader.ts[:-1]])
    kv_cell = np.concatenate([xts_array_cell, k_est_cell, v_est_cell], axis=1)
    kv_cell_df = pd.DataFrame(kv_cell, columns=['x1','x2','t1','t2','density','speed'])
    kv_cell_filepath = os.path.join(output_directory, f'{output_identifier}_kv.csv')
    kv_cell_df.to_csv(kv_cell_filepath, index=False)