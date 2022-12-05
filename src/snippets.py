### Process images - REDUNDANT

# image_size = 800
# crop_size = 512

# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)

# img_list = os.listdir(data_dir)

# # crop to 512x512
# for i in range(len(img_list)):
#     img = Image.open(data_dir + img_list[i])
#     c_x = (img.size[0] - crop_size) // 2
#     c_y = (img.size[1] - crop_size) // 2
#     img = img.crop([c_x, c_y, c_x + crop_size, c_y + crop_size])
#     # img = img.resize((image_size, image_size), Image.Resampling.BILINEAR) # Redundant
#     img.save(preprocessed_dir + img_list[i], 'JPEG')

#     if i % 500 == 0:
#         print('Resizing %d images...' % i)