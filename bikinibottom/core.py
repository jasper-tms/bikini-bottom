#!/usr/bin/env python3

from cloudvolume import CloudVolume

import npimage
import npimage.operations


def compress_raw_cloudvolume(source_path, target_path=None):
    source = CloudVolume(source_path)
    assert source.encoding == 'raw'
    if target_path is None:
        assert source_path.endswith('.raw.ng')
        target_path = source_path.replace('.raw.ng', '.jpeg.ng')
    target_info = CloudVolume.create_new_info(
        num_channels = source.num_channels,
        layer_type = 'image',
        data_type = source.data_type,
        encoding = 'jpeg',
        resolution = source.resolution,
        voxel_offset = source.voxel_offset,
        chunk_size = source.chunk_size,
        volume_size = source.volume_size,
    )

    tprint(f'Compressing {source_path} to {target_path}')
    target = CloudVolume(target_path, info=target_info)
    target.commit_info()

    # Iterate through each chunk and upload it
    for x in range(0, source.shape[0] - source.chunk_size[0], source.chunk_size[0]):
        for y in range(0, source.shape[1] - source.chunk_size[1], source.chunk_size[1]):
            for z in range(0, source.shape[2] - source.chunk_size[2], source.chunk_size[2]):
                chunk = source[x:x+source.chunk_size[0],
                               y:y+source.chunk_size[1],
                               z:z+source.chunk_size[2]]
                target[x:x+source.chunk_size[0],
                       y:y+source.chunk_size[1],
                       z:z+source.chunk_size[2]] = chunk

    # Handle the last row/column/slice
    for dim in range(3):
        dim_size = source.shape[dim]
        chunk_size = source.chunk_size[dim]
        if dim_size % chunk_size != 0:
            start = dim_size - (dim_size % chunk_size)
            slices = [slice(None)] * 3
            slices[dim] = slice(start, dim_size)
            chunk = source[tuple(slices)]
            target[tuple(slices)] = chunk


def downsample_cloudvolume(vol: [CloudVolume, str], data=None, return_downsampled_data=False):
    """
    Downsample the image data in a cloudvolume by a factor of 2 in x, y, and z.

    If no data is provided, the data will be downloaded from the highest
    currently available mip of the volume.
    If data is provided, it should be exactly equal to the data contained in
    the highest currently available mip of the volume, otherwise things will
    get weird.

    If you want to downsample a few times without having to re-download the
    data on each iteration, you can do something like:
    >>> vol.mip = vol.available_mips[-1]
    >>> data = vol[:]
    >>> for iteration in range(num_of_downsamplings):
    >>>     data = downsample_cloudvolume(vol, data=data, return_downsampled_data=True)
    """
    if isinstance(vol, str):
        vol = CloudVolume(vol)
    vol.mip = vol.available_mips[-1]
    print(f'Current mip: {vol.mip}')
    print(f'Will generate mip {vol.mip + 1} = scale {(2**(vol.mip+1),)*3}')
    vol.add_scale((2**(vol.mip+1),)*3, chunk_size=vol.chunk_size)
    vol.commit_info()
    if data is None:
        data = vol[:]
    if not data.shape == vol.shape:
        raise ValueError(f'Expected data to have shape {vol.shape}, but'
                         f' it had shape {data.shape}.')

    data_downsampled = npimage.operations.downsample(data, factor=2)

    vol.mip += 1
    assert vol.mip == vol.available_mips[-1]
    assert vol.shape == data_downsampled.shape
    vol[:] = data_downsampled

    if return_downsampled_data:
        return data_downsampled
