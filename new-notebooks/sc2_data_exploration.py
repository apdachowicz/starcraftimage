#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'lab_black')


# In[2]:


import torch
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
import random


# In[3]:


# load starcraftimage

sys.path.append("..")

from sc2image.dataset import StarCraftImage, StarCraftCIFAR10, StarCraftMNIST

from sc2image.utils.unit_type_data import (
    TERRAN_NAME_TO_ID,
    ZERG_NAME_TO_ID,
    PROTOSS_NAME_TO_ID,
    NEUTRAL_NAME_TO_ID,
    NONNEUTRAL_NAME_TO_ID,
)


# ## Starcraft II Data Exploration
# 
# This notebook downloads and reads SCII images and metadata, following the logic in [this demo notebook](https://github.com/inouye-lab/starcraftimage/blob/main/dataset-demos/starcraftimage-quickstart.ipynb) from the dataset authors (original repo [here](https://github.com/inouye-lab/starcraftimage/tree/main)).
# 
# We load the hyperspectral data in this notebook. Per the authors, each data point in this set contains a 384 x 64 x 64 dimensional array, where the first dimension is the number of unit id channels, and the last two dimensions are the (x,y) dimensions of the map. Each of these data points summarizes 255 frames of a game (about 10 seconds of gameplay), where each pixel in the image is the most recent frame in the stack where the corresponding unit crossed the pixel. 

# #### 1) Describe Image Channels
# 
# Per the source paper, we expect 170 unit channels for each of the two players and 44 channels for neutral units in loaded hyperspectral image data, for a total of 2 * 170 + 44 = 384 image channels.

# In[4]:


print(f"Terran:  {len(TERRAN_NAME_TO_ID)}")
print(f"Zerg:    {len(ZERG_NAME_TO_ID)}")
print(f"Protoss: {len(PROTOSS_NAME_TO_ID)}")
print(
    f"Total player IDs: {len(TERRAN_NAME_TO_ID) + len(ZERG_NAME_TO_ID) + len(PROTOSS_NAME_TO_ID)} (plus 2 additional for no unit or unknown)"
)
print(f"Total non-neutral IDs: {len(NONNEUTRAL_NAME_TO_ID)}")
print(f"Neutral: {len(NEUTRAL_NAME_TO_ID)}")


# Non-neutral (player unit) IDs are ordered:
# - missing / unknown data [0,1] (2 channels)
# - Terran [2, 57] (56 channels)
# - Protoss [58, 99] (42 channels)
# - Zerg [100, 169] (70 channels)

# In[5]:


NONNEUTRAL_NAME_TO_ID


# #### 2) Load Data
# 
# Get train and test data:

# In[6]:


train_data_dir = Path("..") / "data_train"

test_data_dir = Path("..") / "data_test"

for p in [train_data_dir, test_data_dir]:
    if not os.path.exists(p):
        os.makedirs(p)


# In[7]:


# generic args for the data download
image_format = "dense-hyperspectral"
label_kind = "14-class"
return_label = True
return_dict = True


# In[8]:


# download the training data, or load it from disk if already downloaded

print(f"Downloading to: {train_data_dir}")

train_scimage = StarCraftImage(
    train_data_dir,
    image_format=image_format,
    label_kind=label_kind,
    return_label=return_label,
    return_dict=return_dict,
    train=True,
    download=True,
    use_metadata_cache=True,
    verbose=True,
)


# #### 3) Describe Data Format
# 
# Each data point is a length-3 tuple:

# In[9]:


len(train_scimage[0])


# The first element is the 384 x 64 x 64 torch tensor:

# In[10]:


train_scimage[0][0].shape


# The second element is the data label, encoding the map and whether the data comes from the first or second half of the game:

# In[11]:


for label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    print(label)
    sample_count = 0

    for i in range(500):
        if train_scimage[i][1] == label:
            print(
                train_scimage[i][2]["metadata"]["target_id"],
                ":::",
                train_scimage[i][2]["metadata"]["map_name"],
                np.round(
                    train_scimage[i][2]["metadata"]["window_idx"]
                    / train_scimage[i][2]["metadata"]["num_windows"],
                    4,
                ),
                "|||",
                train_scimage[i][2]["metadata"]["player_1_race"],
                train_scimage[i][2]["metadata"]["player_2_race"],
                train_scimage[i][2]["metadata"]["winning_player_id"],
            )

            sample_count += 1

        if sample_count >= 3:
            break

    print()


# The third element is the metadata, and this also includes useful tabular information:

# In[12]:


train_scimage[0][2]


# ##### Quick Data Check
# 
# I was running into an issue where for about 40% of data points, starcraftimage threw an index error when getting the metadata dictionary. I think this was due to missing creep data when neither player is Zerg. I updated `starcraftimage` to return `None` for creep data if this error occurs, and this seems to fix the issue. I will add this fix as a new pull request. 

# In[13]:


bad_indices = []
num_samples = 1000

for i in range(num_samples):
    try:
        train_scimage[i]
    except IndexError:
        bad_indices.append(i)

print(
    f"Indices with IndexError: {len(bad_indices)}, {np.round(len(bad_indices)*100 / num_samples, 3)}% of {num_samples} samples"
)


# #### 4) Check Data
# 
# First, let's check to make sure we observe the expected unit types for each player, given their race.
# 
# It looks like the units are ordered protoss, then terran, then zerg in the hyperspectral data:

# In[14]:


# switch order of protoss and terran units to match to expected?

protoss_names = [x for x in list(NONNEUTRAL_NAME_TO_ID.keys()) if "PROTOSS_" in x]
terran_names = [x for x in list(NONNEUTRAL_NAME_TO_ID.keys()) if "TERRAN_" in x]
zerg_names = [x for x in list(NONNEUTRAL_NAME_TO_ID.keys()) if "ZERG_" in x]

NONNEUTRAL_NAME_TO_ID_CORR = (
    list(NONNEUTRAL_NAME_TO_ID.keys())[:2] + protoss_names + terran_names + zerg_names
)

# get neutral units too
NEUTRAL_NAME_TO_ID_CORR = list(NEUTRAL_NAME_TO_ID.keys())


# In[15]:


# make sure that we get unit data for the expected units only

for data_sample in random.sample(np.arange(50000).tolist(), 3):
    print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    hyp_img = train_scimage[data_sample][0].numpy()

    p1_race = train_scimage[data_sample][2]["metadata"]["player_1_race"]
    p2_race = train_scimage[data_sample][2]["metadata"]["player_2_race"]
    window_id = train_scimage[data_sample][2]["metadata"]["window_idx"]
    total_windows = train_scimage[data_sample][2]["metadata"]["num_windows"]
    map_name = train_scimage[data_sample][2]["metadata"]["map_name"]
    winner = train_scimage[data_sample][2]["metadata"]["winning_player_id"]

    print(map_name)
    print(f"P1: {p1_race}\nP2: {p2_race}\nWinner: {winner}")
    print(f"Window {window_id} of {total_windows}")

    # get mask showing if any pixel has a value for each unit type
    present_ixs = np.any(hyp_img > 0, axis=(1, 2))

    print("\nPLAYER 1 UNITS")
    for i in range(170):
        if present_ixs[i]:
            print(i, NONNEUTRAL_NAME_TO_ID_CORR[i])

    print("\nPLAYER 2 UNITS")
    for i in range(170, 340):
        if present_ixs[i]:
            print(i - 170, NONNEUTRAL_NAME_TO_ID_CORR[i - 170])

    print("\nNEUTRAL UNITS")
    for i in range(340, 344):
        if present_ixs[i]:
            print(i - 340, NEUTRAL_NAME_TO_ID_CORR[i - 340])

    del hyp_img

# present_ixs


# The observed units match the expected units from player race metadata. Let's write a quick check to cycle through many examples and assert that we only get the expected unit types for each player:

# In[16]:


test_size = 2000

for data_sample in range(test_size):
    if data_sample % 500 == 0:
        print(f"Test {data_sample+1} of {test_size}")
    hyp_img = train_scimage[data_sample][0].numpy()

    p1_race = train_scimage[data_sample][2]["metadata"]["player_1_race"]
    p2_race = train_scimage[data_sample][2]["metadata"]["player_2_race"]

    # get mask showing if any pixel has a value for each unit type
    present_ixs = np.any(hyp_img > 0, axis=(1, 2))

    for i in range(170):
        if present_ixs[i]:
            assert (p1_race.upper() + "_" in NONNEUTRAL_NAME_TO_ID_CORR[i]) or (
                "Unknown" in NONNEUTRAL_NAME_TO_ID_CORR[i]
            ), f"Unit {NONNEUTRAL_NAME_TO_ID_CORR[i]} not expected for p1 {p1_race}"

    for i in range(170, 340):
        if present_ixs[i]:
            assert (p2_race.upper() + "_" in NONNEUTRAL_NAME_TO_ID_CORR[i - 170]) or (
                "Unknown" in NONNEUTRAL_NAME_TO_ID_CORR[i]
            ), f"Unit {NONNEUTRAL_NAME_TO_ID_CORR[i-170]} not expected for p2 {p2_race}"

    del hyp_img, present_ixs, p1_race, p2_race

print(f"First {test_size} data points all have only expected player units.")


# #### 5) Visualize Data (WIP)

# In[17]:


def get_map_player_images(
    scimage_sample,
    unit_correspondence=NONNEUTRAL_NAME_TO_ID_CORR + NEUTRAL_NAME_TO_ID_CORR,
):
    hyp_img = scimage_sample[0].numpy()

    p1_race = scimage_sample[2]["metadata"]["player_1_race"]
    p2_race = scimage_sample[2]["metadata"]["player_2_race"]

    # get mask showing if any pixel has a value for each unit type
    present_ixs = np.any(hyp_img > 0, axis=(1, 2))

    # for each player, build corresponding categorical image

    return


# In[ ]:





# In[ ]:




