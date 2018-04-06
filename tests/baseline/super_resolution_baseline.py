from scipy import ndimage, misc
import numpy as np

data_points = 100

folder = "../../data/BSD100_SR/"

for algo in ["nearest", "lanczos", "bilinear", "bicubic", "cubic"]:

    for zoom_level in [2, 3, 4]:

        sub_folder = folder + "image_SRF_" + str(zoom_level) + "/"

        PSNRs = np.zeros(data_points)
        diffs = np.zeros(data_points)

        for i in np.arange(1, data_points + 1):

            filename_prefix = "img_" + \
                "{:03d}".format(i) + "_SRF_" + str(zoom_level) + "_"

            lr_file = sub_folder + filename_prefix + "LR.png"
            hr_file = sub_folder + filename_prefix + "HR.png"

            lr_image = ndimage.imread(lr_file, mode="YCbCr")
            hr_image = ndimage.imread(hr_file, mode="YCbCr")

            h, w, c = hr_image.shape

            upscaled_image = misc.imresize(
                lr_image, hr_image.shape, interp=algo)

            h, w, c = hr_image.shape

            MSE = (1.0 / (h * w)) * \
                np.sum(np.sum((hr_image - upscaled_image)**2, axis=0), axis=0)

            MSE = MSE.mean()

            PSNRs[i - 1] = 20.0 * np.log10(255) - 10.0 * np.log10(MSE)

        print(algo, zoom_level, np.mean(PSNRs))
