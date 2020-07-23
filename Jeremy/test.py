import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import dicom
import pydicom
import os
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

os.getcwd()  # Prints directory

# The training data
train = pd.read_csv('data/train.csv')
train.dtypes
train.describe()

# Function to plot 1 person
def plt_person(id, labs=True):
    f = plt.figure()
    train_id = train[train['Patient'] == id]
    plt.plot(train_id['Weeks'], train_id['FVC'], 'ro-')
    if(labs):
        plt.xlabel('Week')
        plt.ylabel('FVC scores')
    plt.title('Patient ID: {}'.format(id))
    plt.show()

# Example EDA
plt.style.use('seaborn')
plt_person('ID00007637202177411956430')

# Batches of 20
row, col, batch = (5, 4, 2)
IDs = train['Patient'].unique()[row*col*(batch-1):row*col*batch]
fig, axs = plt.subplots(row, col, figsize=(12, 15))
for i in range(row):
    for j in range(col):
        if(col*i+j >= len(IDs)): break
        id = IDs[col*i+j]
        train_id = train[train['Patient'] == id]
        axs[i, j].plot(train_id['Weeks'], train_id['FVC'], 'ro-')
        axs[i, j].set_title('Patient ID: {}'.format(id), fontdict={'fontsize': mpl.rcParams['axes.titlesize']*.7})

# Hide x labels and tick labels for top and right plots
for ax in axs.flat:
    ax.set(xlabel='Week', ylabel='FVC Score')
    ax.label_outer()
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)
plt.tight_layout(h_pad=4.5, w_pad=1.5)
plt.show()

# Image reading
patients = os.listdir('data/train')
paths = ['data/train/' + x for x in patients]

path = 'data/train/' + patients[0]
# Load the scans in given folder path. Stores size thickness
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    # [x.ImagePositionPatient for x in slices] # xyz coords of patient
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2])) # Sorted by y position fo scan
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

[x.SliceThickness for x in load_scan(path)] # All 10
s = load_scan(path)

# Converting to the appropriate HU units
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

get_pixels_hu(s)
