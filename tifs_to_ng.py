#!/usr/bin/env python3
# Based loosely on example code from
# https://github.com/seung-lab/cloud-volume/wiki/Example-Single-Machine-Dataset-Upload

# Usage: ./tifs_to_ng.py [folder] [metadata_file]
# Given the name of a folder (and optionally a metadata file), convert
# all the tifs inside it to neuroglancer precomputed format.
#
# Result: A new dataset will be created at [target_root]/[folder].ng.
# target_root is specified in the script below, so be sure to set it to
# the desired data location. It can be either a local path or a path
# to a google cloud storage bucket.

# The metadata can be provided as a file named metadata.json in the
# same folder as the tifs, or a different filename specified as the second
# command line argument of this script. If no metadata file is provided,
# default metadata hardcoded in this script will be used.
# For any metadata that is required but not provided in the metadata file,
# the default hardcoded values will be used.
#
# If you provide a metadata file it must be in json format,
# with any number of the following keys:
# - owners: a list of names/email addresses/homepage urls of the owners of the dataset
# - description: a string describing the dataset
# - voxel_size_nm: a list of 3 numbers, the size of a voxel in nanometers
# - chunk_size: a list of 3 integers, the size of a chunk in voxels
# - invert: a boolean, whether to invert black and white in the images
# - num_mips: an integer, the number of mips (downsamplings of the image) to generate
# If any are not specified, default values will be used. See the default values
# in the script below.

import sys
import os
import json
import math
from glob import glob
from concurrent.futures import ProcessPoolExecutor


import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import touch
import npimage
import npimage.operations
from tqdm import tqdm

import bikinibottom

# This creates a dataset on google cloud storage. You must have already
# set up the bucket and you must have permissions to write to it (see
# the cloudvolume wiki for instructions on how to set up a bucket).
#target_root = 'gs://your-google-cloud-storage-bucket'
# This creates a dataset on your computer, adjacent to the folder
# containing your tifs
target_root = 'file://.'

default_metadata = dict(
    owners=['Your Name <your.email@foo.com>'],
    description=("No description provided."
                 " Source folder name: {img_folder}"),
    voxel_size_nm=(1, 1, 1),
    encoding='jpeg',
    chunk_size=(128, 128, 128),
    invert=False,
    num_mips=3
)

img_folder = sys.argv[1]
assert os.path.isdir(img_folder), 'First argument is not a folder'
img_filenames = glob(f'{img_folder}/*.tif')
img_filenames.sort()

# Metadata
if len(sys.argv) > 2:
    metadata_fn = sys.argv[2]
    if not os.path.isfile(metadata_fn):
        raise FileNotFoundError(f'Metadata file not found: {metadata_fn}')
else:
    metadata_fn = os.path.join(img_folder, 'metadata.json')
metadata = default_metadata.copy()
# Open the metadata file (json format), and update the default metadata with
# whatever is in the file.
if not os.path.isfile(metadata_fn):
    print('WARNING: Default metadata will be used since metadata.json was'
          ' not found in the data folder.')
else:
    with open(metadata_fn, 'r') as f:
        metadata.update(json.load(f))
if '{img_folder}' not in metadata['description']:
    metadata['description'] = metadata['description'] + " Source folder name: {img_folder}"
print('Metadata:')
print(json.dumps(metadata, indent=2))

# Determine source data properties
shape_z = len(img_filenames)
first_im = npimage.open(img_filenames[0], dim_order='xy')
shape_x, shape_y = first_im.shape
shape = (shape_x, shape_y, shape_z)
source_dtype = first_im.dtype

# Create a new cloudvolume
info = CloudVolume.create_new_info(
    num_channels = 1,
    layer_type = 'image', # 'image' or 'segmentation'
    data_type = 'uint8', # can pick any popular uint
    encoding = metadata['encoding'], # other options: 'jpeg', 'compressed_segmentation' (req. uint32 or uint64)
    resolution = metadata['voxel_size_nm'], # X,Y,Z values in nanometers
    voxel_offset = [0, 0, 0], # values X,Y,Z values in voxels
    chunk_size = metadata['chunk_size'], # rechunk of image X,Y,Z in voxels
    volume_size = shape, # X,Y,Z size in voxels
)
target_path = target_root + '/' + img_folder.rstrip('/') + '.' + metadata['encoding'] + '.ng'
print(f'Opening a cloudvolume at {target_path}')
vol = CloudVolume(target_path, info=info) #, parallel=8)
vol.provenance.description = metadata['description'].format(img_folder=img_folder)
vol.provenance.owners = metadata['owners']
vol.commit_info() # generates gs://bucket/dataset/info json file
vol.commit_provenance() # generates gs://bucket/dataset/provenance json


# Load image data from a series of tifs
data = np.zeros(shape + (1,), dtype=source_dtype)
for z, fn in enumerate(tqdm(img_filenames)):
    data[:, :, z, 0] = npimage.open(fn, dim_order='xy')
if data.dtype == np.uint16:
    print('Converting from uint16 to uint8')
    try:
        clip_range = metadata['8bit_range']
    except:
        # If clip range is not specified, npimage.operations.to_8bit will
        # by default use the 0.05th percentile and the 99.95th percentile, which
        # is reasonable
        clip_range = [None, None]
    data = npimage.operations.to_8bit(data, bottom_value=clip_range[0], top_value=clip_range[1])
if not data.dtype == np.uint8:
    raise ValueError(f'Expected data to be uint8, but it was {data.dtype}')

if metadata.get('invert', False):
    print('Inverting black and white')
    data = 255 - data


# Upload the data to the cloudvolume
vol[:] = data[:]
# Generate downsampling levels if requested
for mip in range(metadata['num_mips']):
    print(f'Downsampling to mip {mip+1} of {metadata["num_mips"]}')
    data = bikinibottom.downsample_cloudvolume(vol, data=data, return_downsampled_data=True)
