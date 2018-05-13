from PIL import Image, ImageDraw

test_folder = "../../tests/test_figures/patch_generation/"

file_num = "253027"

filenames = ["_0_1_nearest", "_Bi_HR_4",
             "_5_HR-out", "_GT", "_Lap_HR_4"]

for filename in filenames:
    filename = file_num + filename
    img = Image.open("images/{}.png".format(filename))

    W, H = img.size
    print(filename, W, H)

    x1 = 110
    y1 = 161
    x2 = 172
    y2 = 205

    bbox = (x1, y1, x2, y2)

    img_cropped = img.crop(bbox)

    W1, H1 = img_cropped.size

    x = W // W1

    imsize = (W, x * H1)
    img_cropped = img_cropped.resize(imsize, Image.BICUBIC)
    img_cropped.save("generated_images/{}_cropped.png".format(filename))

    dr = ImageDraw.Draw(img)

    f = "red"
    w = 5

    line = (bbox[0], bbox[1], bbox[0], bbox[3])
    dr.line(line, fill=f, width=w)

    line = (bbox[0], bbox[1], bbox[2], bbox[1])
    dr.line(line, fill=f, width=w)

    line = (bbox[0], bbox[3], bbox[2], bbox[3])
    dr.line(line, fill=f, width=w)

    line = (bbox[2], bbox[1], bbox[2], bbox[3])
    dr.line(line, fill=f, width=w)

    img.save("generated_images/{}_highlighted.png".format(filename))
