import os
from PIL import Image

# TODO: change this individually
data_dir = '../data/'
save_dir = '../data_preprocessed/preprocessed_64/'

image_size = 64
crop_size = 512

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

img_list = os.listdir(data_dir)

for i in range(5000):
    img = Image.open(data_dir + img_list[i])
    c_x = (img.size[0] - crop_size) // 2
    c_y = (img.size[1] - crop_size) // 2
    img = img.crop([c_x, c_y, c_x + crop_size, c_y + crop_size])
    img = img.resize((image_size, image_size), Image.BILINEAR)
    img.save(save_dir + img_list[i], 'JPEG')

    if i % 1000 == 0:
        print('Resizing %d images...' % i)