import numpy as np
import matplotlib.pyplot as plt

factor = 4

folder = "../../data/BSDS300/images/test/"
test_folder = "../../tests/test_figures/deep_image_prior/" + \
    str(factor) + "x/"

test_folder_mse = test_folder + "latest/"
test_folder_char = test_folder + "skip 4x 2k_it char_loss ssim/"

filenames = [16077, 37073, 101085, 189080,
             21077, 85048, 86000, 106024, 210088, 253027]


psnr_history_mse = None
psnr_history_char = None

for i, f in enumerate(filenames):

    filename_mse = test_folder_mse + "{}__psnr_history.npy".format(f)
    if i == 0:
        psnr_history_mse = np.load(filename_mse)
    else:
        psnr_history_mse += np.load(filename_mse)

    filename_char = test_folder_char + "{}__psnr_history.npy".format(f)
    if i == 0:
        psnr_history_char = np.load(filename_char)
    else:
        psnr_history_char += np.load(filename_char)


print()

its, _ = psnr_history_mse.shape

leng = len(filenames)

psnr_history_mse /= leng
psnr_history_char /= leng

t = np.arange(its)

fig = plt.figure()
plt.title('PSNR v/s Iterations', fontsize=16)
plt.xlabel('Iterations', fontsize=16)
plt.ylabel('PSNR', fontsize=16)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=16)
# plt.plot(t, psnr_history_mse[:, 0].reshape(None), color='r', label="PSNR_MSE_LR")
plt.plot(t, psnr_history_mse[:, 1].reshape(None), color='b', label="PSNR MSE loss")
# plt.plot(t, psnr_history_char[:, 0].reshape(None), color='g', label="PSNR_CHAR_LR")
plt.plot(t, psnr_history_char[:, 1].reshape(None), color='r', label="PSNR Charbonnier loss")
# k: black

plt.legend()

plt.grid(linestyle='--')

plt.show()
