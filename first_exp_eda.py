
# coding: utf-8

# In[136]:


import os
import random
import collections

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import skimage
from skimage import exposure, img_as_float
#plt.style.use('dark_background')
#plt.style.use('seaborn-deep')
plt.style.use('Solarize_Light2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[86]:


print(plt.style.available)


# In[2]:


IMAGE_DIMENSIONS_NUM = 3
images_dir = '../input/train'
segmentation_file_path = '../input/train_ship_segmentations.csv'
full_cwd_path = os.getcwd()
path_prefix, cwd_itself = os.path.split(full_cwd_path)
if cwd_itself != 'code':
    os.chdir(os.path.join(path_prefix, 'code'))
    print(os.getcwd())


# In[3]:


train_images_names = os.listdir(images_dir)


# In[4]:


#len(train_images_names)
print("train_images_names len: {}".format(len(train_images_names)))
#print("train_images_masks_names len: {}".format(len(train_images_masks_names)))
print(train_images_names[:10]) # Is an image name and mask name with the same index math?
segmentation_df = pd.read_csv(segmentation_file_path)
#print(train_images_masks_names[:10])
print(segmentation_df.head())
print(segmentation_df.shape)


# In[5]:


masked_images_df = segmentation_df[~segmentation_df.EncodedPixels.isna()]
no_masked_images_df = segmentation_df[segmentation_df.EncodedPixels.isna()]
#print(len(masked_images_df))
#print(len(no_masked_images_df))
ships_images_num = masked_images_df.shape[0]
no_ships_images_num = no_masked_images_df.shape[0]
print(ships_images_num)
print(no_ships_images_num)


# In[6]:


#train_images_names_sample = masked_images_df['ImageId'][:27000].tolist()
ships_train_images_names = masked_images_df['ImageId'].tolist()
no_ships_train_imasges_names = no_masked_images_df['ImageId'].tolist()
#print(type(ships_train_images_names_sample))
print(ships_train_images_names[:10])


# In[7]:


def select_objects(indexes_list, objects_names):
    return tuple(objects_names[i] for i in indexes_list)


# In[8]:


def select_random_indexses_subset(size, subset_size):
    return random.sample(tuple(range(size)), subset_size)


# In[9]:


def random_objects_select(objects_names, subset_size):
    objects_names_len = len(objects_names)
    indexes = select_random_indexses_subset(objects_names_len, subset_size)
    return select_objects(indexes, objects_names)


# In[10]:


def select_offset_indexses_subset(size, subset_size, offset):
    return tuple(range(size))[offset:offset + subset_size]


# In[11]:


def offset_objects_select(objects_names, subset_size, offset):
    objects_names_len = len(objects_names)
    indexes = select_offset_indexses_subset(objects_names_len, subset_size, offset)
    return select_objects(indexes, objects_names)


# In[134]:


def show_images_grid(images_list, columns_num, figsize, images_names_list=None):
    rows_num = len(images_list) // columns_num
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize)
    images_list_len = len(images_list)
    slice_len = columns_num
    images_list_slices_len = images_list_len // slice_len
    images_names_list_slices = []
    images_list_slices = [
        images_list[i * slice_len:(i+1) * slice_len]\
        for i in range(images_list_slices_len)
    ]
    if images_names_list:
        images_names_list_slices = [
            images_names_list[i * slice_len:(i+1) * slice_len]\
            for i in range(images_list_slices_len)
        ]
    #print("show_images_grid, images_names_list_slices:\n", images_names_list_slices)
    for i in range(ax.shape[0]):
        images_list_slice = images_list_slices[i]
        images_names_list_slice = []
        if images_names_list:
            images_names_list_slice = images_names_list_slices[i]
        #images_list_0 = [images_pair[0] for images_pair in paired_images_list_slice]
        #images_list_1 = [images_pair[1] for images_pair in paired_images_list_slice]
        for j in range(columns_num):
            if images_names_list:
                #print("show_images_grid, images_names_list_slice[{}]:\n".format(j), images_names_list_slice[j])
                ax[i, j].set_title(images_names_list_slice[j])            
            ax[i, j].imshow(images_list_slice[j])

            #ax[2 * i + 1, j].imshow(images_list_1[j])
    plt.tight_layout()
    plt.show()


# In[100]:


def draw_images_with_histograms(images_list, columns_num, figsize, nbins):
    rows_num = 3 * len(images_list) // columns_num
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize)
    images_list_len = len(images_list)
    slice_len = columns_num
    images_list_slices_len = images_list_len // slice_len
    images_list_slices = [
    images_list[i * slice_len:(i+1) * slice_len]\
        for i in range(images_list_slices_len)
    ]
    for i in range(ax.shape[0] // 3):
        images_list_slice = images_list_slices[i]
        for j in range(columns_num):
            ax[3 * i, j].imshow(images_list_slice[j])
            ax[3 * i, j].grid(False)
            img_cdf, cdf_bins = exposure.cumulative_distribution(images_list_slice[j].flatten())
            ax[3 * i + 1, j].hist(exposure.histogram(images_list_slice[j].flatten(), nbins=nbins))
            ax[3 * i + 2, j].plot(cdf_bins, img_cdf, color='green')
    plt.show()


# In[14]:


def extract_object_pixels_by_mask(image, mask):
    return image * mask
apply_mask_to_color_channel = extract_object_pixels_by_mask


# In[15]:


def extract_not_object_pixels_by_mask(image, mask):
    #return image * (np.ones(mask.shape) - mask)
    not_object_pixels = image * (np.ones(mask.shape) - mask)
    #print("extract_non_salt_pixels_by_mask:\n", non_salt_pixels)
    return not_object_pixels
apply_negative_mask_to_color_channel = extract_not_object_pixels_by_mask


# In[16]:


def get_mask_complement(mask):
    return np.ones(mask.shape) - mask


# In[17]:


def compute_object_pixels_num(mask):
    return mask.sum()

def compute_not_object_pixels_num(mask):
    return mask.size - mask.sum()


# In[18]:


def compute_whole_image_mean_brightnes(image):
    return image.mean()


# In[155]:


def compute_color_channel_mean_brightness(color_channel):
    return color_channel.mean()


# In[156]:


def compute_color_channels_mean_brightness(color_channels):
    return [
        compute_color_channel_mean_brightness(color_channel) for color_channel in color_channels
    ]


# In[157]:


def compute_images_list_color_channels_mean_brightness(images_color_channels_list):
    return [
        compute_color_channels_mean_brightness(
            image_color_channels
        ) for image_color_channels in images_color_channels_list
    ]


# In[158]:


def divide_color_channels_values(color_channels_values):
    return (
        [color_channel_value[0] for color_channel_value in color_channels_values],
        [color_channel_value[1] for color_channel_value in color_channels_values],
        [color_channel_value[2] for color_channel_value in color_channels_values],
    )


# In[19]:


def compute_object_pixels_mean_brightness(image, mask):
    full_color_image_salt_pixels = apply_mask_to_full_color_image(image, mask)
    #print(compute_salt_pixels_num(mask))
    salt_pixels_num = compute_salt_pixels_num(mask) * IMAGE_DIMENSIONS_NUM
    salt_pixels_sum = full_color_image_salt_pixels.sum()
    return 0.0 if salt_pixels_num == 0 else salt_pixels_sum / salt_pixels_num


# In[20]:


def get_image_color_channels(image):
    return [image[:, :, i] for i in range(IMAGE_DIMENSIONS_NUM)]

def collapse_color_channels_to_image(color_components):
    return np.array([color_components[i].T for i in range(IMAGE_DIMENSIONS_NUM)]).T


# In[21]:


def get_images_color_channels(images):
    return [get_image_color_channels(image) for image in images]


# In[22]:


def apply_mask_to_color_channels(color_channels, mask):
    return [apply_mask_to_color_channel(color_channel, mask) for color_channel in color_channels]


# In[23]:


def apply_negative_mask_to_color_channels(color_channels, mask):
    #print("apply_negative_mask_to_color_channels, color_channels:\n", color_channels)
    return [apply_negative_mask_to_color_channel(color_channel, mask) for color_channel in color_channels]


# In[24]:


def apply_mask_to_full_color_image(image, mask):
    color_channels = get_image_color_channels(image)
    masked_color_channels = apply_mask_to_color_channels(
        color_channels,
        mask
    )
    return collapse_color_channels_to_image(masked_color_channels)


# In[25]:


def apply_negative_mask_to_full_color_image(image, mask):
    color_channels = get_image_color_channels(image)
    masked_color_channels = apply_negative_mask_to_color_channels(
        color_channels,
        mask
    )
    return collapse_color_channels_to_image(masked_color_channels)


# In[219]:


def calculate_mode(sample):
    #print("calculate_mode, sample.shape:", sample.shape)
    #print(sample)
    flatten_sample = sample.flatten()
    #print("calculate_mode, flatten_sample.shape:", flatten_sample.shape)
    #print("calculate_mode, flatten_sample:\n", flatten_sample)
    counter = collections.Counter(sample.flatten())
    #print("calculate_mode, counter:\n", counter)
    most_common = counter.most_common()
    if len(most_common) > 0:
        if len(most_common) == 1:
            return most_common[0]
        else:
            if most_common[0][0] == 0:
                return most_common[1]
            else:
                return most_common[0]
    else:
        return None
compute_color_channel_mode = calculate_mode


# In[232]:


def collect_modes_and_frequencies(modes):
    modes_frequencies_dict = {}
    print("collect_modes_and_frequencies,  modes:\n", modes)
    for color_channels_modes in modes:
        mode_value, frequency = color_channels_modes[0]
        if mode_value in modes_frequencies_dict:
            modes_frequencies_dict[mode_value] += frequency
        else:
            modes_frequencies_dict[mode_value] = frequency
    return modes_frequencies_dict


# In[234]:


def gather_color_channel_modes_and_frequencies(modes):
    modes_frequencies_dict = {}
    #print("collect_modes_and_frequencies,  modes:\n", modes)
    for color_channels_modes in modes:
        mode_value, frequency = color_channels_modes
        if mode_value in modes_frequencies_dict:
            modes_frequencies_dict[mode_value] += frequency
        else:
            modes_frequencies_dict[mode_value] = frequency
    return modes_frequencies_dict


# In[194]:


def gather_objects_and_frequencies(objects_list):
    return collections.Counter(objects_list)


# In[199]:


def gather_objects_and_frequencies(objects_list):
    frequencies_dict = {}
    for obj in objects_list:
        if obj in frequencies_dict:
            frequencies_dict[obj] += 1
        else:
            frequencies_dict[obj] = 1
    return frequencies_dict


# In[171]:


def sort_gathered_objects(objects):
    objects_sorted = sorted(objects)
    objects_frequencies_sorted = []
    for obj in objects_sorted:
        objects_frequencies_sorted.append(objects[obj])
    return objects_sorted, objects_frequencies_sorted


# In[28]:


def compute_color_channels_modes(color_channels):
    #print("compute_color_channels_modes, len(color_channels): ", len(color_channels))
    #print("compute_color_channels_modes, color_channels:\n", color_channels)
    return [compute_color_channel_mode(color_channel) for color_channel in color_channels]


# In[29]:


def compute_images_color_channels_modes(images_color_channels):
    return [
        compute_color_channels_modes(image_color_channels)\
        for image_color_channels in images_color_channels
    ]


# In[30]:


def get_image_pixels(color_channels):
    return [
        (channel0value, channel1value, channel2value) for channel0value, channel1value, channel2value
           in zip(color_channels[0], color_channels[1], color_channels[2])
    ]


# In[31]:


def get_image_pixels(image):
    color_channel0 = image[:, :, 0].ravel()
    color_channel1 = image[:, :, 1].ravel()
    color_channel2 = image[:, :, 2].ravel()
    return [(x, y, z) for x, y, z in zip(color_channel0, color_channel1, color_channel2)]


# In[32]:


def compare_color_channels_by_pixels(image):
    image_pixels = get_image_pixels(image)
    return [compare_pixel_color_channels_values(pixel) for pixel in image_pixels]


# In[33]:


def compare_images_color_channels(images):
    return [compare_color_channels_by_pixels(image) for image in images]


# In[34]:


ships_train_images_names_sample = random_objects_select(ships_train_images_names, int(ships_images_num / 9))
no_ships_train_images_names_sample = random_objects_select(no_ships_train_imasges_names, int(no_ships_images_num / 9))


# In[37]:


#train_images = [mpimg.imread(os.path.join(images_dir, image_name)) for image_name in train_images_names]
ships_train_images = [
    mpimg.imread(os.path.join(images_dir, image_name)) for image_name in ships_train_images_names_sample
]


# In[38]:


no_ships_train_images = [
    mpimg.imread(os.path.join(images_dir, image_name)) for image_name in no_ships_train_images_names_sample
]


# In[39]:


plt.imshow(ships_train_images[0])
plt.show()
#plt.imshow(train_images_masks[0])
#plt.show()


# In[40]:


plt.imshow(ships_train_images[3471])
plt.show()
#plt.imshow(train_images_masks[0])
#plt.show()


# In[41]:


plt.imshow(no_ships_train_images[0])
plt.show()


# In[42]:


plt.imshow(no_ships_train_images[3471])
plt.show()


# In[43]:


ships_images_subsample = ships_train_images[:40]


# In[44]:


no_ships_images_subsample = no_ships_train_images[:40]


# In[137]:


show_images_grid(ships_images_subsample, 8, (24, 24), images_names_list=ships_train_images_names_sample[:40])


# In[138]:


show_images_grid(no_ships_images_subsample, 8, (24, 24))


# In[139]:


draw_images_with_histograms(ships_images_subsample[:40], 8, (24, 24), 20)


# In[140]:


draw_images_with_histograms(no_ships_images_subsample[:40], 8, (24, 24), 20)


# In[49]:


ships_train_images_mean_brightness = [
    compute_whole_image_mean_brightnes(image) for image in ships_train_images
]


# In[50]:


no_ships_train_images_mean_brightness = [
    compute_whole_image_mean_brightnes(image) for image in no_ships_train_images
]


# In[141]:


fig = plt.figure(figsize=(24, 24))
ax = fig.add_subplot(111)
ax.hist(ships_train_images_mean_brightness, bins=300)
ax.set_title("Mean brightnes distribution for images with chips")
#ax.set(title="Mean brightnes distribution for images with chips")
ax.set_xlabel("Mean brightness")
ax.set_ylabel("Num of images")
plt.show()


# In[142]:


fig = plt.figure(figsize=(24, 24))
ax = fig.add_subplot(111)
ax.hist(no_ships_train_images_mean_brightness, bins=300)
ax.set_title("Mean brightnes distribution for no chips images")
ax.set_xlabel("Mean brightness")
ax.set_ylabel("Num of images")
plt.show()


# In[53]:


#random_selected_train_images = random_objects_select(train_images, 9000)


# In[54]:


#print(type(random_selected_train_images[0]))
#print(random_selected_train_images[0])


# In[55]:


#random_selected_train_images_mean_brightness = [
#    compute_whole_image_mean_brightnes(image) for image in random_selected_train_images
#]


# In[56]:


#fig = plt.figure(figsize=(24, 24))
#ax = fig.add_subplot(111)
#ax.hist(random_selected_train_images_mean_brightness, bins=100)
#plt.show()


# In[58]:


#ships_train_images[0]


# In[96]:


print(ships_train_images_mean_brightness[0])


# In[97]:


print(no_ships_train_images_mean_brightness[0])


# In[98]:


print((sum(no_ships_train_images_mean_brightness)) / len(no_ships_train_images_mean_brightness))


# In[99]:


print((sum(ships_train_images_mean_brightness)) / len(ships_train_images_mean_brightness))


# In[107]:


ships_train_images_color_channels = [get_image_color_channels(image) for image in ships_train_images]


# In[108]:


no_ships_train_images_color_channels = [get_image_color_channels(image) for image in no_ships_train_images]


# In[109]:


ships_train_images_color_channels_modes = compute_images_color_channels_modes(
    ships_train_images_color_channels
)
no_ships_train_images_color_channels_modes = compute_images_color_channels_modes(
    no_ships_train_images_color_channels
)


# In[111]:


ships_train_images_color_channels_modes_collected = collect_modes_and_frequencies(
    ships_train_images_color_channels_modes
)
no_ships_train_images_color_channels_modes_collected = collect_modes_and_frequencies(
    no_ships_train_images_color_channels_modes
)


# In[125]:


ships_train_images_modes_values = ships_train_images_color_channels_modes_collected.keys()
ships_train_images_modes_frequencies = ships_train_images_color_channels_modes_collected.values()
no_ships_train_images_modes_values = no_ships_train_images_color_channels_modes_collected.keys()
no_ships_train_images_modes_frequencies = no_ships_train_images_color_channels_modes_collected.values()


# In[129]:


ships_train_images_modes_values_sorted = sorted(ships_train_images_modes_values)
ships_train_images_modes_frequencies_sorted = []
for mode_value in ships_train_images_modes_values_sorted:
    ships_train_images_modes_frequencies_sorted.append(ships_train_images_color_channels_modes_collected[mode_value])

no_ships_train_images_modes_values_sorted = sorted(no_ships_train_images_modes_values)
no_ships_train_images_modes_frequencies_sorted = []
for mode_value in no_ships_train_images_modes_values_sorted:
    no_ships_train_images_modes_frequencies_sorted.append(no_ships_train_images_color_channels_modes_collected[mode_value])


# In[143]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.hist(no_ships_train_images_modes_values, bins=200)
plt.show()


# In[144]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.hist(no_ships_train_images_modes_frequencies, bins=70)
plt.show()


# In[145]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.hist(ships_train_images_modes_values, bins=200)
plt.show()


# In[146]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.hist(ships_train_images_modes_modes_frequencies, bins=70)
plt.show()


# In[151]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.set_title("No ships train images brighthess modes values/modes frequencies")
ax.plot(no_ships_train_images_modes_values_sorted, no_ships_train_images_modes_frequencies_sorted)
#ax.legend()
plt.show()


# In[148]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(ships_train_images_modes_values_sorted, ships_train_images_modes_frequencies_sorted)
plt.show()


# In[154]:


print(type(ships_train_images_color_channels))
print(len(ships_train_images_color_channels))
print(len(ships_train_images_color_channels[0]))
print(type(ships_train_images_color_channels[0]))
print(ships_train_images_color_channels[0])
print()


# In[160]:


images_list_color_channels_mean_brightness = compute_images_list_color_channels_mean_brightness(
    no_ships_train_images_color_channels
)


# In[179]:


print(type(images_list_color_channels_mean_brightness))
print(images_list_color_channels_mean_brightness[0])


# In[161]:


divided_to_color_channels_images_mean_brightness = divide_color_channels_values(
    images_list_color_channels_mean_brightness    
)


# In[190]:


print(type(divided_to_color_channels_images_mean_brightness))
print(len(divided_to_color_channels_images_mean_brightness))
print(type(divided_to_color_channels_images_mean_brightness[0]))
print(len(divided_to_color_channels_images_mean_brightness[0]))
print(len(divided_to_color_channels_images_mean_brightness[1]))
print(len(divided_to_color_channels_images_mean_brightness[2]))
print(type(divided_to_color_channels_images_mean_brightness[0][0]))
#print(len(divided_to_color_channels_images_mean_brightness[0][0]))
print(divided_to_color_channels_images_mean_brightness[0][0])
print(divided_to_color_channels_images_mean_brightness[1][0])
print(divided_to_color_channels_images_mean_brightness[2][0])
#print(
#    divided_to_color_channels_images_mean_brightness[0],
#    divided_to_color_channels_images_mean_brightness[1],
#    divided_to_color_channels_images_mean_brightness[2]
#)


# In[166]:


no_ship_images_red_color_channel_mean_brightness = divided_to_color_channels_images_mean_brightness[0]
no_ship_images_yellow_color_channel_mean_brightness = divided_to_color_channels_images_mean_brightness[1]
no_ship_images_blue_color_channel_mean_brightness = divided_to_color_channels_images_mean_brightness[2]


# In[200]:


gathered_no_ship_images_red_color_channel_mean_brightness = gather_objects_and_frequencies(
    no_ship_images_red_color_channel_mean_brightness
)


# In[168]:


gathered_no_ship_images_yellow_color_channel_mean_brightness = gather_objects_and_frequencies(
    no_ship_images_yellow_color_channel_mean_brightness
)


# In[169]:


gathered_no_ship_images_blue_color_channel_mean_brightness = gather_objects_and_frequencies(
    no_ship_images_blue_color_channel_mean_brightness
)


# In[202]:


gathered_no_ship_images_red_color_channel_mean_brightness_sorted = sort_gathered_objects(
    gathered_no_ship_images_red_color_channel_mean_brightness
)


# In[203]:


no_ship_images_red_color_channel_mean_brightness_sorted = gathered_no_ship_images_red_color_channel_mean_brightness_sorted
no_ship_images_red_color_channel_mean_brightness_frequencies_sorted = gathered_no_ship_images_red_color_channel_mean_brightness_sorted


# In[204]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(
    no_ship_images_red_color_channel_mean_brightness_sorted,
    no_ship_images_red_color_channel_mean_brightness_frequencies_sorted,
    color='red',
    label='red',
    lw=2
)
plt.show()


# In[177]:


print(type(gathered_no_ship_images_red_color_channel_mean_brightness))
print(len(gathered_no_ship_images_red_color_channel_mean_brightness))


# In[205]:


print(type(ships_train_images_color_channels_modes))
print(len(ships_train_images_color_channels_modes))


# In[206]:


print(ships_train_images_color_channels_modes[0])


# In[209]:


print(type(ships_train_images_color_channels))
print(len(ships_train_images_color_channels))
print(ships_train_images_color_channels[0])


# In[240]:


no_ships_tr_imgs_clr_ch_modes = [
    compute_color_channels_modes(
        img_clr_ch_modes
    ) for img_clr_ch_modes in no_ships_train_images_color_channels
]


# In[210]:


ships_tr_imgs_clr_ch_modes = [
    compute_color_channels_modes(
        img_clr_ch_modes
    ) for img_clr_ch_modes in ships_train_images_color_channels
]


# In[224]:


print(type(ships_tr_imgs_clr_ch_modes))
print(len(ships_tr_imgs_clr_ch_modes))
print(ships_tr_imgs_clr_ch_modes[0])
print(ships_tr_imgs_clr_ch_modes[1])
print(ships_tr_imgs_clr_ch_modes[2])


# In[225]:


compute_color_channel_mode(ships_train_images_color_channels[0][0])


# In[226]:


#print(type(ships_train_images_color_channels_modes_collected))
#print(len(ships_train_images_color_channels_modes_collected))
#print(ships_train_images_color_channels_modes_collected)


# In[241]:


no_ships_tr_imgs_clr_ch_modes_div = divide_color_channels_values(no_ships_tr_imgs_clr_ch_modes)
ships_tr_imgs_clr_ch_modes_div = divide_color_channels_values(ships_tr_imgs_clr_ch_modes)


# In[242]:


print(type(ships_tr_imgs_clr_ch_modes_div))
print(len(ships_tr_imgs_clr_ch_modes_div))


# In[243]:


no_ships_tr_imgs_clr_ch_modes_red = gather_color_channel_modes_and_frequencies(no_ships_tr_imgs_clr_ch_modes_div[0])
no_ships_tr_imgs_clr_ch_modes_yellow = gather_color_channel_modes_and_frequencies(no_ships_tr_imgs_clr_ch_modes_div[1])
no_ships_tr_imgs_clr_ch_modes_blue = gather_color_channel_modes_and_frequencies(no_ships_tr_imgs_clr_ch_modes_div[2])


# In[244]:


ships_tr_imgs_clr_ch_modes_red = gather_color_channel_modes_and_frequencies(ships_tr_imgs_clr_ch_modes_div[0])
ships_tr_imgs_clr_ch_modes_yellow = gather_color_channel_modes_and_frequencies(ships_tr_imgs_clr_ch_modes_div[1])
ships_tr_imgs_clr_ch_modes_blue = gather_color_channel_modes_and_frequencies(ships_tr_imgs_clr_ch_modes_div[2])


# In[239]:


print(type(ships_tr_imgs_clr_ch_modes_red))
print(len(ships_tr_imgs_clr_ch_modes_red))
print(type(ships_tr_imgs_clr_ch_modes_yellow))
print(len(ships_tr_imgs_clr_ch_modes_yellow))
print(type(ships_tr_imgs_clr_ch_modes_blue))
print(len(ships_tr_imgs_clr_ch_modes_blue))


# In[ ]:


#ships_train_images_modes_values_sorted = sorted(ships_train_images_modes_values)
#ships_train_images_modes_frequencies_sorted = []
#for mode_value in ships_train_images_modes_values_sorted:
#    ships_train_images_modes_frequencies_sorted.append(ships_train_images_color_channels_modes_collected[mode_value])


# In[245]:


no_ships_tr_imgs_clr_ch_modes_and_freq_r_sr = sort_gathered_objects(no_ships_tr_imgs_clr_ch_modes_red)
no_ships_tr_imgs_clr_ch_modes_and_freq_y_sr = sort_gathered_objects(no_ships_tr_imgs_clr_ch_modes_yellow)
no_ships_tr_imgs_clr_ch_modes_and_freq_b_sr = sort_gathered_objects(no_ships_tr_imgs_clr_ch_modes_blue)


# In[246]:


ships_tr_imgs_clr_ch_modes_and_freq_r_sr = sort_gathered_objects(ships_tr_imgs_clr_ch_modes_red)
ships_tr_imgs_clr_ch_modes_and_freq_y_sr = sort_gathered_objects(ships_tr_imgs_clr_ch_modes_yellow)
ships_tr_imgs_clr_ch_modes_and_freq_b_sr = sort_gathered_objects(ships_tr_imgs_clr_ch_modes_blue)


# In[247]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(
    no_ships_tr_imgs_clr_ch_modes_and_freq_r_sr[0],
    no_ships_tr_imgs_clr_ch_modes_and_freq_r_sr[1],
    color='red'
)
plt.show()


# In[248]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(
    ships_tr_imgs_clr_ch_modes_and_freq_r_sr[0],
    ships_tr_imgs_clr_ch_modes_and_freq_r_sr[1],
    color='red'
)
plt.show()


# In[249]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(
    no_ships_tr_imgs_clr_ch_modes_and_freq_y_sr[0],
    no_ships_tr_imgs_clr_ch_modes_and_freq_y_sr[1],
    color='yellow'
)
plt.show()


# In[250]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(
    ships_tr_imgs_clr_ch_modes_and_freq_y_sr[0],
    ships_tr_imgs_clr_ch_modes_and_freq_y_sr[1],
    color='yellow'
)
plt.show()


# In[251]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(
    no_ships_tr_imgs_clr_ch_modes_and_freq_b_sr[0],
    no_ships_tr_imgs_clr_ch_modes_and_freq_b_sr[1],
    color='blue'
)
plt.show()


# In[252]:


fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(111)
ax.plot(
    ships_tr_imgs_clr_ch_modes_and_freq_b_sr[0],
    ships_tr_imgs_clr_ch_modes_and_freq_b_sr[1],
    color='blue'
)
plt.show()

