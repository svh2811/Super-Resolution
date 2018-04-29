from __future__ import print_function
import matplotlib.pyplot as plt

import argparse
import os

import numpy as np
from models import *

import torch
import torch.optim

import time

from skimage.measure import compare_psnr
from models.downsampler import Downsampler

from utils.sr_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

factor = 2  # 2 or 3 or 4
PLOT = True

folder = "../../data/BSDS300/images/test/"
test_folder = "../../tests/test_figures/deep_image_prior/" + \
    str(factor) + "x/"

input_depth = 32

INPUT = 'noise'
pad = 'reflection'
OPT_OVER = 'net'
KERNEL_TYPE = 'lanczos2'

LR = 0.01
tv_weight = 0.0

OPTIMIZER = 'adam'

if factor == 2:
    num_iter = 150 + 1
    reg_noise_std = 0.01
elif factor == 4:
    num_iter = 2000 + 1
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 8000 + 1
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'

plot_freq = 50

# skip, UNet, ResNet
NET_TYPE = 'skip'

# filenames = [16077, 37073, 101085, 189080,
#             21077, 85048, 86000, 106024, 210088, 253027]

filenames = [16077]

num_files = len(filenames)
psnr_sum = 0.0

print("Factor: ", factor, " | Images: ", num_files,
      " | Iterations: ", num_iter, "\n\n")


print(
    "filename, psnr_nearest, psnr_bicubic, psnr_history[-1, :], final_psnr_HR, time_diff")

for filename in filenames:
    plot_prefix = test_folder + str(filename) + "_"
    fname = folder + str(filename) + ".jpg"
    imgs = load_LR_HR_imgs_sr(fname, -1, factor, enforse_div32="CROP")

    # print('HR and LR resolutions: %s, %s' % (str(img['HR_pil'].size), str(img['LR_pil'].size)))

    imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] \
        = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

    img = np_to_pil(imgs['nearest_np'])
    img.save(plot_prefix + "0_1_nearest.png")

    img = np_to_pil(imgs['bicubic_np'])
    img.save(plot_prefix + "0_2_bicubic.png")

    img = np_to_pil(imgs['bicubic_np'])
    img.save(plot_prefix + "0_3_sharp.png")

    img = imgs['HR_pil']
    img.save(plot_prefix + "0_4_HR_GT.png")

    if PLOT:
        plot_image_grid([imgs['nearest_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['HR_np']],
                        nrow=4,
                        factor=19,
                        save_plot=True,
                        fname=plot_prefix + "1_nearest_bicubic_sharp_HR.png")

        psnr_bicubic = compare_psnr(imgs['HR_np'], imgs['bicubic_np'])
        psnr_nearest = compare_psnr(imgs['HR_np'], imgs['nearest_np'])

        # print('PSNR bicubic: %.4f   PSNR nearest: %.4f' % (psnr_bicubic, psnr_nearest))

    net_input = get_noise(input_depth,
                          INPUT,
                          (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

    net = get_net(input_depth,
                  NET_TYPE,
                  pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)

    img_LR_var = np_to_var(imgs['LR_np']).type(dtype)

    downsampler = Downsampler(n_planes=3,
                              factor=factor,
                              kernel_type=KERNEL_TYPE,
                              phase=0.5,
                              preserve_size=True).type(dtype)

    def closure():
        global i

        if reg_noise_std > 0:
            net_input.data = net_input_saved + \
                (noise.normal_() * reg_noise_std)

        out_HR = net(net_input)
        out_LR = downsampler(out_HR)

        total_loss = mse(out_LR, img_LR_var)

        if tv_weight > 0:
            total_loss += tv_weight * tv_loss(out_HR)

        total_loss.backward()

        # Log
        psnr_LR = compare_psnr(imgs['LR_np'], var_to_np(out_LR))
        psnr_HR = compare_psnr(imgs['HR_np'], var_to_np(out_HR))
        # print('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f'
        # % (i, psnr_LR, psnr_HR), '\r', end='')

        # History
        psnr_history[i, :] = [psnr_LR, psnr_HR]

        if PLOT and i % plot_freq == 0:
            out_HR_np = var_to_np(out_HR)
            plot_image_grid([imgs['bicubic_np'], imgs['HR_np'], np.clip(out_HR_np, 0, 1)],
                            nrow=3,
                            factor=20,
                            save_plot=True,
                            fname=plot_prefix + "2_" + "{:05d}".format(i) + "_bicubic_HR_out.png")

        i += 1

        return total_loss

    psnr_history = np.zeros((num_iter, 2))
    net_input_saved = net_input.data.clone()
    noise = net_input.data.clone()

    i = 0
    p = get_params(OPT_OVER, net, net_input)

    # print('Starting optimization with ' + str(OPTIMIZER))
    start = time.time()
    optimize(OPTIMIZER, p, closure, LR, num_iter)
    time_diff = time.time() - start

    out_HR_np = var_to_np(net(net_input))
    final_psnr_HR = compare_psnr(imgs['HR_np'], out_HR_np)
    out_HR_np = np.clip(out_HR_np, 0, 1)

    psnr_sum += final_psnr_HR

    result_deep_prior = put_in_center(out_HR_np, imgs['HR_np'].shape[1:])

    plot_image_grid([imgs['nearest_np'], imgs['bicubic_np'], out_HR_np, imgs['HR_np']],
                    nrow=4,
                    factor=19,
                    save_plot=True,
                    fname=plot_prefix + "3.png")

    print(filename, psnr_nearest, psnr_bicubic,
          psnr_history[-1, :], final_psnr_HR, time_diff)

    # os.system('spd-say
    # \"U-Net with Skip Connections has increased resolutions of 1 images\"')

    np.save(plot_prefix + "_psnr_history", psnr_history)

    t = np.arange(num_iter)

    fig = plt.figure(num="psnr_LR and psnr_HR",
                     figsize=(23, 12))
    ax = fig.add_subplot(111)

    ax.plot(t,
            psnr_history[:, 0].reshape(None),
            c='r',
            ls='-',
            lw=0.5,
            label='psnr_LR')

    ax.plot(t,
            psnr_history[:, 1].reshape(None),
            c='b',
            ls='-',
            lw=0.5,
            label='psnr_HR')

    plt.xlabel('Iterations')
    plt.ylabel('PSNR')

    plt.legend(loc=2)

    plt.savefig(fname=plot_prefix + "4.png",
                transparent=True,
                bbox_inches='tight',
                pad_inches=0)

    plt.close('all')

    img = np_to_pil(out_HR_np)
    img.save(plot_prefix + "5_HR-out.png")

    img = np_to_pil(result_deep_prior)
    img.save(plot_prefix + "5_DP-res.png")


print("\n\nAverage psnr for {} images : {}".format(num_files,
                                                   psnr_sum / num_files))

# os.system('spd-say \"Job Super Resolution completed\"')
