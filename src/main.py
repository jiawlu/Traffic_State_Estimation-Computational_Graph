# @author       Jiawei Lu (jiaweil9@asu.edu)
# @time         2021/3/19 12:10
# @desc         [script description]
import os

from load_data import *
from net import Model
from trainer import Trainer
from save_results import *



variant = dict(
    data_directory=r'data\MobileCentury',
    output_directory=r'data\MobileCentury\outputs',


    space_start=0,  # meter
    space_end=5000,
    time_start=000,
    time_end=7200,  # second
    grid_space_length=100,
    grid_time_length=60,

    x_scale=0.001,
    t_scale=0.001,
    density_scale=10,
    speed_scale=0.01,

    use_loop=True,
    use_gps=True,
    use_bluetooth=False,
    use_video_ave_density=False,
    use_travel_time=True,

    loop_weight=1,
    gps_weight=1,
    bluetooth_weight=0.0001,
    video_ave_density_weight=1,
    travel_time_weight=1e-7,
    mu_weight=1e-2,
    pde_weight=0.0001,
    vk_model_weight=0.001,
    max_q_weight=1e-2,

    number_of_epochs=10000,
    loss_eval_interval=1,
    number_of_batches=3,

    net_kwargs=dict(
        shared_hidden_sizes=[125, 125, 125],
        hidden_sizes_1=[125, 125, 125, 125, 125, 125],        # k
        hidden_sizes_2=[125, 125, 125, 125, 125, 125]),       # v
    learning_rate=0.001,
    amsgrad=False
)
variant['volume_scale'] = variant['density_scale'] * variant['speed_scale']


def run():
    data_loader = DataLoader(variant)
    model = Model(**variant['net_kwargs'], density_scale=variant['density_scale'], speed_scale=variant['speed_scale'])
    trainer = Trainer(data_loader, model)
    trainer.train()
    outputResults(trainer)


if __name__ == '__main__':
    run()

