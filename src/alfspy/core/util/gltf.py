import io
from typing import Optional
from urllib.request import urlopen

import numpy as np
from gltflib import GLTF
from numpy.typing import NDArray

from alfspy.core.rendering.data import MeshData, TextureData
from alfspy.core.geo.transform import Transform
from alfspy.core.util.image import bytes_to_img


def _get_from_buffer(idx: int, gltf: GLTF, comp: int, dtype: str = 'f4') -> NDArray:
    """
    Interprets the portion of the first buffer of the given ``GLTF`` object
    associated with the given buffer view index as a vector with given amount of components.
    :param idx: Buffer view index of the data to be read.
    :param gltf: The ``GLTF`` object holding all data.
    :param comp: The amount of components of the vectors to read.
    :param dtype: The type the read values should be converted to.
    :return: A numpy array representing all read vectors.
    """
    buffer = gltf.resources[0].data
    accessor = list(filter(lambda acc: acc.bufferView == idx, gltf.model.accessors))[0]
    buf_view = gltf.model.bufferViews[accessor.bufferView]
    data_bytes = buffer[buf_view.byteOffset:buf_view.byteOffset + buf_view.byteLength]
    return np.frombuffer(data_bytes, dtype=dtype).reshape((-1, comp))


def gltf_to_mesh_data(gltf: GLTF, transform: Optional[Transform] = None) -> Optional[MeshData]:
    """
    Extracts the first mesh found within the ``GLTF`` object.
    :param gltf: The ``GLTF`` object to extract the mesh from.
    :param transform: The transform to add to the extracted mesh data (optional).
    :return: If the ``GLTF`` object contains no meshes returns ``None``;
    otherwise returns a ``MeshData`` object containing vertices, indices and UV coordinates.
    If indices and/or UV coordinates cannot be extracted they will be set to None.
    """
    if len(gltf.model.meshes) < 1 or len(gltf.model.meshes[0].primitives) < 1:
        return None
    mesh_attrs = gltf.model.meshes[0].primitives[0].attributes

    vertices = _get_from_buffer(mesh_attrs.POSITION, gltf, 3)

    indices_idx = gltf.model.meshes[0].primitives[0].indices
    indices = _get_from_buffer(indices_idx, gltf, 3, dtype='u4') if indices_idx is not None else None

    uvs_idx = mesh_attrs.TEXCOORD_0
    uvs = _get_from_buffer(uvs_idx, gltf, 2) if uvs_idx is not None else None

    return MeshData(vertices, indices, uvs, transform)


def gltf_to_texture_data(gltf: GLTF) -> Optional[TextureData]:
    """
    Extracts the texture of the main mesh found within the ``GLTF`` object.
    :param gltf: The ``GLTF`` object to extract the texture from.
    :return: If the there is no mesh within the ``GLTF`` object or the main mesh has no texture returns ``None``;
    otherwise returns a ``TextureData`` object containing the extracted texture.
    """

    if len(gltf.model.meshes) < 1 or len(gltf.model.meshes[0].primitives) < 1:  # No main mesh exists
        return None

    material_idx = gltf.model.meshes[0].primitives[0].material  # Main mesh has no material
    if material_idx is None:
        return None

    material = gltf.model.materials[material_idx]
    texture_info = material.pbrMetallicRoughness.baseColorTexture
    if texture_info is None or texture_info.index is None:  # Material has no texture assigned
        return None

    # assume texture is embedded using a base64 URI
    uri = gltf.model.images[texture_info.index].uri

    with urlopen(uri) as resp:
        data = resp.read()
    texture = bytes_to_img(data)
    texture = texture[::-1, ...]  # texture is extracted horizontally flipped
    return TextureData(texture) if texture is not None else None


def gltf_extract(file: str, transform: Optional[Transform] = None) -> tuple[Optional[MeshData], Optional[TextureData]]:
    """
    Extracts mesh and texture data from a GLTF file via ``get_mesh_data`` and ``get_texture_data``.
    :param file: Path to a GLTF file.
    :param transform: The transform to add to the extracted mesh data (optional).
    :return: A tuple containing the results of ``get_mesh_data`` and ``get_texture_data``.
    """
    gltf_data = GLTF.load(file, load_file_resources=True)
    return gltf_to_mesh_data(gltf_data, transform), gltf_to_texture_data(gltf_data)

def glb_extract_from_bytes(data: bytes, transform: Optional[Transform] = None)-> \
        tuple[Optional[MeshData], Optional[TextureData]]:
    """
    Extracts mesh and texture data from GLB byte data via ``get_mesh_data`` and ``get_texture_data``.
    :param data: Bytes representing GLB data.
    :param transform: The transform to add to the extracted mesh data (optional).
    :return: A tuple containing the results of ``get_mesh_data`` and ``get_texture_data``.
    """
    data_stream = io.BytesIO(data)
    gltf_data = GLTF.read_glb(data_stream, load_file_resources=False)
    return gltf_to_mesh_data(gltf_data, transform), gltf_to_texture_data(gltf_data)

def gltf_extract_from_bytes(data: bytes, transform: Optional[Transform] = None)-> \
        tuple[Optional[MeshData], Optional[TextureData]]:
    """
    Extracts mesh and texture data from GLTF byte data via ``get_mesh_data`` and ``get_texture_data``.
    :param data: Bytes representing GLTF data.
    :param transform: The transform to add to the extracted mesh data (optional).
    :return: A tuple containing the results of ``get_mesh_data`` and ``get_texture_data``.
    """
    data_stream = io.BytesIO(data)
    gltf_data = GLTF.read_gltf(data_stream, load_file_resources=False)
    return gltf_to_mesh_data(gltf_data, transform), gltf_to_texture_data(gltf_data)