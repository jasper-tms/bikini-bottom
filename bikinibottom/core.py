#!/usr/bin/env python3

import os
from typing import Union, Optional

import numpy as np
import skimage
import trimesh
from cloudvolume import CloudVolume
import cloudvolume.exceptions
import cloudvolume.mesh
import npimage


def compress_raw_cloudvolume(source_path: str,
                             target_path: Optional[str] = None):
    """
    Make a copy of a raw (uncompressed or lossless compressed) cloudvolume
    that is jpeg compressed (lossily compressed).

    Parameters
    ----------
    source_path : str
        The path to the source cloudvolume. This should be a path to a
        cloudvolume that is raw (uncompressed or lossless compressed).

    target_path : str, optional
        The path to the target cloudvolume. If not provided, the target
        cloudvolume will be saved in the same directory as the source
        cloudvolume, with the same name, but with the extension changed to
        ".jpeg.ng".
    """
    source = CloudVolume(source_path)
    assert source.encoding == 'raw'
    if target_path is None:
        if source_path.endswith('.raw.ng'):
            target_path = source_path.replace('.raw.ng', '.jpeg.ng')
        elif source_path.endswith('.ng'):
            target_path = source_path.replace('.ng', '.jpeg.ng')
        else:
            target_path = source_path + '.jpeg.ng'
    target_info = CloudVolume.create_new_info(
        num_channels=source.num_channels,
        layer_type='image',
        data_type=source.data_type,
        encoding='jpeg',
        resolution=source.resolution,
        voxel_offset=source.voxel_offset,
        chunk_size=source.chunk_size,
        volume_size=source.volume_size,
    )

    print(f'Compressing {source_path} to {target_path}')
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


def downsample_cloudvolume(vol: Union[str, CloudVolume],
                           data: Optional[np.ndarray] = None,
                           compress: bool = True,
                           return_downsampled_data: bool = False) -> Optional[np.ndarray]:
    """
    Downsample the image data in a cloudvolume by a factor of 2 in x, y, and z.

    If the data argument is left as None, the cloudvolume's data will be
    downloaded from the highest currently available mip (lowest resolution)
    of the volume.
    If data is provided, it should be exactly equal to the data contained in
    the highest currently available mip of the volume, otherwise things will
    get weird.
    The intended use of the data argument is if you want to downsample a few
    times without having to re-download the data from the cloud on each
    iteration, you should do something like:
    >>> vol.mip = vol.available_mips[-1]
    >>> data = vol[:]
    >>> for iteration in range(num_of_downsamplings):
    >>>     data = downsample_cloudvolume(vol, data=data, return_downsampled_data=True)
    """
    original_mip = None
    if isinstance(vol, str):
        vol = CloudVolume(vol, compress=compress)
    elif hasattr(vol, 'mip'):
        original_mip = vol.mip
    vol.mip = vol.available_mips[-1]
    print(f'Current mip: {vol.mip}')
    print(f'Will generate mip {vol.mip + 1} = scale {(2**(vol.mip+1),)*3}')
    try:
        vol.add_scale((2**(vol.mip+1),)*3, chunk_size=vol.chunk_size)
        vol.commit_info()
        if data is None:
            data = vol[:]
        if not data.shape == vol.shape:
            raise ValueError(f'Expected data to have shape {vol.shape}, but'
                             f' it had shape {data.shape}.')

        data_downsampled = npimage.downsample(data, factor=2)

        vol.mip += 1
        assert vol.mip == vol.available_mips[-1]
        assert vol.shape == data_downsampled.shape
        vol[:] = data_downsampled
    finally:
        if original_mip is not None:
            vol.mip = original_mip

    if return_downsampled_data:
        return data_downsampled


def mesh_array(data: np.ndarray,
               threshold: float,
               discard_small_components: bool = False,
               save_to_filename: Optional[str] = None) -> Union[trimesh.Trimesh, None]:
    """
    Generate a mesh from a numpy array.

    Parameters
    ----------
    data : np.ndarray
        The array to mesh.
    threshold : float
        The threshold to use for the marching cubes algorithm.
    discard_small_components : bool, default False
        If True, only the single largest connected component of the mesh
        will be returned. Otherwise, the entire mesh will be returned.
    save_to_filename : str, optional
        If provided, the mesh will be saved to this file. Otherwise, the
        mesh will be returned.

    Returns
    -------
    If save_to_filename is None, the mesh will be returned. Otherwise,
    saves the mesh to the provided filename and returns None.
    """
    verts, faces, _, _ = skimage.measure.marching_cubes(data, threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if discard_small_components:
        components = mesh.split(only_watertight=False)
        mesh = max(components, key=lambda component: len(component.faces))
    if save_to_filename is None:
        return mesh
    else:
        mesh.export(save_to_filename)


def mesh_cloudvolume(vol: Union[str, CloudVolume],
                     threshold: float,
                     mip: Optional[int] = None,
                     discard_small_components: bool = False,
                     save_to_filename: Optional[str] = None) -> Union[trimesh.Trimesh, None]:
    """
    Generate a mesh from a cloudvolume.

    Parameters
    ----------
    vol : CloudVolume or str
        The cloudvolume to mesh. If a string is provided, it will be
        interpreted as the path to a cloudvolume.
    threshold : float
        The pixel intensity threshold to use for the marching cubes algorithm.
    mip : int, optional
        The mip level to use. If not provided, the highest available mip
        (lowest resolution data) will be used.
    discard_small_components : bool, default False
        If True, only the single largest connected component of the mesh
        will be returned. Otherwise, the entire mesh will be returned.
    save_to_filename : str, optional
        If provided, the mesh will be saved to this file. Otherwise, the
        mesh will be returned.

    Returns
    -------
    If save_to_filename is None, the mesh will be returned. Otherwise,
    saves the mesh to the provided filename and returns None.
    """
    original_mip = None
    if isinstance(vol, str):
        vol = CloudVolume(vol)
    elif hasattr(vol, 'mip'):
        original_mip = vol.mip
    if mip is None:
        mip = vol.available_mips[-1]
    vol.mip = mip
    try:
        data = np.array(vol[:].squeeze())
        mesh = mesh_array(data, threshold,
                          discard_small_components=discard_small_components,
                          save_to_filename=save_to_filename)
        if save_to_filename is not None:
            return save_to_filename
        # Scale vertex coordinates to use physical units (usually nanometers)
        mesh.vertices = mesh.vertices * vol.resolution

    finally:
        if original_mip is not None:
            vol.mip = original_mip
    return mesh


def push_mesh(mesh: Union[str, trimesh.Trimesh, cloudvolume.mesh.Mesh],
              mesh_id: int,
              vol: Union[str, CloudVolume],
              scale_by: float = 1,
              compress: bool = True,
              overwrite: bool = False,
              mesh_subfolder: str = 'mesh') -> None:
    """
    Upload a mesh representing the outline of some bit of an image volume
    to a cloudvolume that can be loaded alongside that image volume.

    If the given CloudVolume is a segmentation volume, the mesh will be
    uploaded directly to it. If the given volume is an image volume, a new
    segmentation-type CloudVolume will be created in a subfolder of that image
    cloudvolume named "meshes/" and the mesh will be uploaded there.

    Parameters
    ----------
    mesh : str or a mesh object (Trimesh, DracoMesh, cloudvolume.mesh, etc)
        If a string is provided, it will be interpreted as the path to
        a mesh file on your computer and will be loaded using trimesh.load().
        Otherwise, it must be a mesh object with both:
        1) either a .vertices or .points attribute
        2) a .faces attribute

    mesh_id : int
        The id of the mesh. This is the segment ID that will need to be entered
        into neuroglancer to load the mesh.

    vol : str or CloudVolume
        The cloudvolume to upload the mesh to. If a string is provided, it will
        be interpreted as the path to a cloudvolume.

    scale_by : float, default 1
        The scale factor to apply to the mesh before uploading it. This is
        useful if the mesh was generated from a downsampled version of the
        image volume, but you want to upload it to the full resolution image
        volume. The default is 1, which means no scaling will be applied.

    compress : bool, default True
        Whether or not to gzip the mesh file, reducing disk usage and network
        bandwidth at the cost of a small amount of compute time.

    overwrite : bool, default False
        Whether or not to overwrite an existing mesh with the same mesh_id.
    """

    if isinstance(vol, str):
        vol = CloudVolume(vol)
    if vol.layer_type == 'image':
        try:
            vol = CloudVolume(vol.cloudpath + '/' + mesh_subfolder)
        except cloudvolume.exceptions.InfoUnavailableError:
            info = CloudVolume.create_new_info(
                num_channels=vol.num_channels,
                layer_type='segmentation',
                mesh=mesh_subfolder,
                data_type='uint8',
                encoding='raw',
                resolution=vol.resolution,
                voxel_offset=vol.voxel_offset,
                # Since we won't be storing any segmentation data here, just
                # make one big chunk to reduce the number of requests that
                # neuroglancer would make for segmentation data.
                chunk_size=vol.volume_size,
                volume_size=vol.volume_size,
            )
            vol = CloudVolume(vol.cloudpath + '/' + mesh_subfolder, info=info)
            vol.commit_info()

    assert vol.layer_type == 'segmentation'
    if not overwrite:
        if vol.mesh.exists([mesh_id], progress=False)[str(mesh_id)]:
            raise FileExistsError(f'A mesh with id {mesh_id} already exists in'
                                  f' the volume {vol.cloudpath}. If you want to'
                                  ' overwrite it, set overwrite=True.')

    if isinstance(mesh, str):
        if os.path.exists(mesh):
            mesh = trimesh.load(mesh)
        else:
            raise FileNotFoundError(f'The file {mesh} does not exist.')

    if hasattr(mesh, 'vertices'):
        vertices = mesh.vertices
    elif hasattr(mesh, 'points'):
        vertices = mesh.points
    else:
        raise ValueError('The mesh provided does not have a "vertices"'
                         ' or "points" attribute.')

    mesh = cloudvolume.mesh.Mesh(
        vertices * scale_by,
        mesh.faces,
        segid=mesh_id
    )
    vol.mesh.put(mesh, compress=compress)
    if 'mesh' not in vol.info:
        vol.info['mesh'] = mesh_subfolder
        vol.commit_info()
