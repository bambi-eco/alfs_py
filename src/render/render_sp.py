import json
from pathlib import Path
from typing import Optional, Iterable

import moderngl as mgl
import numpy as np

from src.core.rendering.camera import Camera
from src.core.rendering.data import TextureData, MeshData
from src.core.rendering.renderer import Renderer
from src.core.rendering.shot import CtxShot
from src.core.rendering.shot_loader import AsyncShotLoader
from src.core.sharepoint.sharepoint_client import SharepointClient
from src.core.sharepoint.sharepoint_shot import SharepointCtxShot
from src.core.util.basic import get_aabb
from src.core.util.gltf import glb_extract_from_bytes, gltf_extract_from_bytes
from src.core.util.image import bytes_to_img
from src.render.data import IntegralSettings, BaseSettings
from src.render.render import make_done_callback, _ensure_or_copy_settings, DoneCallback, make_mgl_context, \
    process_render_data, make_camera, _integral_processing, release_all


def make_sharepoint_client(config_file: str):
    """
    Creates a sharepoint client. Ensures all examples use the same way of initializing a sharepoint client
    :param config_file: The location of the config file containing all data required for client creation
    :return: A sharepoint client instance
    """
    with open(config_file) as f:
        config = json.load(f)

    client_id = config["client_id"]
    client_secret = config["client_secret"]
    tenant_id = config["tenant_id"]
    site = config["site"]
    return SharepointClient(client_id, client_secret, tenant_id, site)


def make_sp_shot_loader(shots: Iterable[CtxShot]) -> Iterable[CtxShot]:
    """
    Creates a shot loader. Ensures all examples use the same parameters when using a shot loader.
    :return: A shot loader instance ensuring the ``Iterable[CtxShot]`` type.
    """
    return AsyncShotLoader(shots, 128, 8)


def read_sp_gltf(gltf_file: str, spc: SharepointClient) -> tuple[MeshData, TextureData]:
    """
    Reads a GLTF file from the sharepoint and transforms it into mesh and texture data that can be used by the renderer
    :param gltf_file: The relative server path pointing at the GLTF file to process. Only supports GLTF and GLB files.
    :param spc: The sharepoint client to be used for retrieving the GLTF file
    :return: The extracted mesh and texture data
    """
    gltf_bytes = spc.get_file_bytes_by_path(gltf_file)
    file_type = Path(gltf_file).suffix.lower()

    if file_type == '.glb':
        return glb_extract_from_bytes(gltf_bytes)
    elif file_type == '.gltf':
        return gltf_extract_from_bytes(gltf_bytes)
    else:
        raise ValueError(f'Invalid file extension {file_type} used')


def read_sp_shots(json_file: str, spc: SharepointClient, ctx: mgl.Context, se: BaseSettings):
    """
    Reads the shots from a JSON file and filters them according to the given settings
    :param json_file: The relative server path of the shot JSON file
    :param spc: The sharepoint client to be used for retrieving the JSON file
    :param ctx: The context the shots should be associated with
    :param se: The basic render settings to be used
    :return: A list of context related shots
    """
    shots = SharepointCtxShot.from_sharepoint_json(json_file, spc, ctx, count=se.count + se.initial_skip,
                                                   correction=se.correction, lazy=se.lazy)
    return shots[se.initial_skip::se.skip]


def read_sp_mask(mask_file: str, spc: SharepointClient) -> TextureData:
    """
    Reads a binary mask from the given file path and converts it into texture data
    :param mask_file: The mask file to be used
    :param spc: The sharepoint client to be used for retrieving the mask
    :return: Texture data resembling the given mask
    """
    mask_bytes = spc.get_file_bytes_by_path(mask_file)
    mask_img = bytes_to_img(mask_bytes)
    mask_img = mask_img[..., 0].astype('f4')
    mask_img = np.resize(mask_img, (*mask_img.shape, 1))
    mask_img /= 255.0
    return TextureData(mask_img)


def _base_steps(done: DoneCallback, scp: SharepointClient, gltf_file: str, shot_json_file: str, mask_file: Optional[str],
                se: BaseSettings) -> tuple[mgl.Context, Camera, Renderer, list[CtxShot], Optional[TextureData]]:
    print(f'  Reading GLTF file from "{gltf_file}"')
    mesh_data, texture_data = read_sp_gltf(gltf_file, scp)
    mesh_data, texture_data = process_render_data(mesh_data, texture_data)
    done()

    print('  Creating MGL context')
    ctx = make_mgl_context()
    done()

    print(f'  Extracting shots from "{shot_json_file}" (Creating lazy shots: {se.lazy})')
    shots = read_sp_shots(shot_json_file, scp, ctx, se)
    done()

    print(f'  Creating camera and renderer (camera position mode: {se.camera_position_mode.name})')
    mesh_aabb = get_aabb(mesh_data.vertices)
    camera = make_camera(mesh_aabb, shots, se)
    print(f'    Computed camera position: {camera.transform.position}')
    renderer = Renderer(se.resolution, ctx, camera, mesh_data, texture_data)
    done()

    if mask_file is not None:
        print(f'  Reading mask from "{mask_file}"')
        mask = read_sp_mask(mask_file, scp)
        done()
    else:
        mask = None

    return ctx, camera, renderer, shots, mask


def render_integral_sp(config_file: str, gltf_file: str, shot_json_file: str, mask_file: Optional[str] = None,
                    settings: Optional[IntegralSettings] = None) -> None:
    done = make_done_callback()
    print('Start projection process')

    # region Initializing

    print('  Initializing')
    se = _ensure_or_copy_settings(settings, IntegralSettings)
    done()

    # endregion

    # region Create sharepoint client

    print('  Creating Sharepoint client')
    scp = make_sharepoint_client(config_file)
    done()

    # endregion

    ctx, _, renderer, shots, mask = _base_steps(done, scp, gltf_file, shot_json_file, mask_file, se)

# region Rendering Integral

    print(f'  Projecting shots (Releasing shots after projection: {se.release_shots})')
    shot_loader = make_sp_shot_loader(shots)
    result = renderer.render_integral(shot_loader, mask=mask, save=False, release_shots=se.release_shots)
    done()

    # endregion

    _integral_processing(done, renderer, result, se)

    print('  Release all resources')
    release_all(ctx, renderer, shots)
    done()

    done.total(msg='All done', indent=False)
