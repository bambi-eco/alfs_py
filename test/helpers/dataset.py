"""Generates a complete miniature ALFS flight on disk.

Everything the production pipelines read is synthesised here: a textured DEM as ``.gltf``,
a matched-poses JSON, a correction JSON, a projection mask, per-frame images and MOT labels.
No real flight data is required to run the integration tests, and because the scene geometry
is chosen (a flat DEM at a known height, cameras on a known path) the expected world-space
scale of every render is analytically known.
"""

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from test.helpers.scenes import fovy_covering

__all__ = ['SyntheticFlight', 'write_gltf', 'build_flight']

# GLTF component/target constants.
_FLOAT = 5126
_UNSIGNED_INT = 5125
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


def _data_uri(payload: bytes, mime: str) -> str:
    return f'data:{mime};base64,{base64.b64encode(payload).decode("ascii")}'


def write_gltf(path: str, vertices: NDArray, indices: NDArray, uvs: NDArray,
               texture_rgb: NDArray) -> str:
    """
    Writes a minimal textured GLTF that :func:`alfspy.core.util.gltf.gltf_extract` can read.

    The layout is deliberately plain -- one buffer, one buffer view per accessor with a zero
    accessor offset, and the texture embedded as a base64 PNG data URI -- because that is
    exactly the shape ``gltf_extract`` assumes. Anything cleverer (interleaved attributes,
    GLB buffer-view images) would not survive its reader.

    :param path: Destination ``.gltf`` path.
    :param vertices: ``(V, 3)`` float32 positions.
    :param indices: ``(T, 3)`` uint32 triangle indices.
    :param uvs: ``(V, 2)`` float32 UV coordinates.
    :param texture_rgb: ``(H, W, 3)`` uint8 RGB texture.
    :return: The path written.
    """
    vertices = np.ascontiguousarray(vertices, dtype='<f4')
    uvs = np.ascontiguousarray(uvs, dtype='<f4')
    indices = np.ascontiguousarray(indices, dtype='<u4').reshape(-1)

    offsets = []
    padded = []
    cursor = 0
    for chunk in (vertices.tobytes(), uvs.tobytes(), indices.tobytes()):
        offsets.append(cursor)
        # Keep every view 4-byte aligned, as the spec requires.
        padding = (-len(chunk)) % 4
        padded.append(chunk + b'\x00' * padding)
        cursor += len(chunk) + padding

    buffer_bytes = b''.join(padded)

    ok, png = cv2.imencode('.png', cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('Failed to encode the synthetic texture as PNG')

    model = {
        'asset': {'version': '2.0', 'generator': 'alfs_pytorch synthetic test fixture'},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'mesh': 0}],
        'buffers': [{
            'byteLength': len(buffer_bytes),
            'uri': _data_uri(buffer_bytes, 'application/octet-stream'),
        }],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': offsets[0], 'byteLength': vertices.nbytes,
             'target': _ARRAY_BUFFER},
            {'buffer': 0, 'byteOffset': offsets[1], 'byteLength': uvs.nbytes,
             'target': _ARRAY_BUFFER},
            {'buffer': 0, 'byteOffset': offsets[2], 'byteLength': indices.nbytes,
             'target': _ELEMENT_ARRAY_BUFFER},
        ],
        'accessors': [
            {'bufferView': 0, 'byteOffset': 0, 'componentType': _FLOAT, 'count': len(vertices),
             'type': 'VEC3',
             'min': vertices.min(axis=0).tolist(), 'max': vertices.max(axis=0).tolist()},
            {'bufferView': 1, 'byteOffset': 0, 'componentType': _FLOAT, 'count': len(uvs),
             'type': 'VEC2'},
            {'bufferView': 2, 'byteOffset': 0, 'componentType': _UNSIGNED_INT,
             'count': int(indices.size), 'type': 'SCALAR'},
        ],
        'images': [{'uri': _data_uri(png.tobytes(), 'image/png')}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9729, 'wrapS': 10497, 'wrapT': 10497}],
        'textures': [{'sampler': 0, 'source': 0}],
        'materials': [{
            'pbrMetallicRoughness': {
                'baseColorTexture': {'index': 0},
                'metallicFactor': 0.0,
                'roughnessFactor': 1.0,
            }
        }],
        'meshes': [{
            'primitives': [{
                'attributes': {'POSITION': 0, 'TEXCOORD_0': 1},
                'indices': 2,
                'material': 0,
            }]
        }],
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(model, handle)
    return path


@dataclass
class SyntheticFlight:
    """
    Paths and ground truth for a generated flight.

    :cvar root: The directory everything was written to.
    :cvar dem_file: The DEM mesh path.
    :cvar poses_file: The matched-poses JSON path.
    :cvar correction_file: The correction JSON path.
    :cvar mask_file: The projection mask path.
    :cvar frame_files: Per-frame image paths, index-aligned with the poses.
    :cvar mot_file: The MOT ground-truth path.
    :cvar dem_height: The constant Z height of the (flat) DEM.
    :cvar camera_height: The Z height every capture was taken from.
    :cvar fovy: The vertical field of view of every capture, in degrees.
    :cvar frame_size: The ``(width, height)`` of the source frames.
    :cvar footprint_half: Half the world-space extent a single capture covers.
    :cvar positions: The camera XY positions per frame.
    :cvar objects: Ground objects as ``(track_id, world_x, world_y)``.
    """
    root: str
    dem_file: str
    poses_file: str
    correction_file: str
    mask_file: str
    frame_files: list = field(default_factory=list)
    mot_file: str = ''
    dem_height: float = 0.0
    camera_height: float = 30.0
    fovy: float = 50.0
    frame_size: tuple = (64, 64)
    footprint_half: float = 0.0
    positions: list = field(default_factory=list)
    objects: list = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        """
        :return: How many frames the flight contains.
        """
        return len(self.frame_files)


def build_flight(root: str,
                 frame_count: int = 5,
                 frame_size: tuple = (64, 64),
                 dem_resolution: int = 12,
                 dem_half: float = 60.0,
                 dem_height: float = 0.0,
                 camera_height: float = 30.0,
                 footprint_half: float = 20.0,
                 step: float = 4.0,
                 flight_id: str = '001',
                 objects: Optional[Sequence[tuple]] = None,
                 seed: int = 7) -> SyntheticFlight:
    """
    Generates a complete flight on disk.

    The DEM is flat at ``dem_height`` and every camera looks straight down from
    ``camera_height`` with a FOV chosen so each frame covers exactly
    ``2 * footprint_half`` world units. That makes the world position of any source pixel
    exact, so label round-trips can be asserted numerically rather than visually.

    Cameras march along ``+X`` in increments of ``step``, so consecutive frames overlap
    heavily -- which is what gives the light-field integration a non-trivial overlap count.

    :param root: Directory to write into.
    :param frame_count: How many frames to generate.
    :param frame_size: The ``(width, height)`` of each source frame.
    :param dem_resolution: Quads per side of the DEM grid.
    :param dem_half: Half the DEM's side length in world units.
    :param dem_height: The DEM's constant Z height.
    :param camera_height: The Z height every capture is taken from.
    :param footprint_half: Half the world extent each capture covers.
    :param step: World-space spacing between consecutive camera positions along X.
    :param flight_id: Prefix used in generated file names.
    :param objects: Ground objects as ``(track_id, world_x, world_y)`` (optional).
    :param seed: Seed for the generated frame patterns.
    :return: A :class:`SyntheticFlight` describing what was written.
    """
    os.makedirs(root, exist_ok=True)
    frames_dir = os.path.join(root, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    width, height = frame_size

    # ── DEM ──────────────────────────────────────────────────────────────────
    n = dem_resolution + 1
    axis = np.linspace(-dem_half, dem_half, n, dtype='f4')
    grid_x, grid_y = np.meshgrid(axis, axis, indexing='xy')
    grid_z = np.full_like(grid_x, dem_height)
    vertices = np.stack((grid_x, grid_y, grid_z), axis=-1).reshape(-1, 3)

    unit = np.linspace(0.0, 1.0, n, dtype='f4')
    grid_u, grid_v = np.meshgrid(unit, unit, indexing='xy')
    uvs = np.stack((grid_u, grid_v), axis=-1).reshape(-1, 2)

    faces = []
    for row in range(dem_resolution):
        for col in range(dem_resolution):
            bl = row * n + col
            br = bl + 1
            tl = bl + n
            tr = tl + 1
            faces.append((bl, br, tr))
            faces.append((bl, tr, tl))
    indices = np.array(faces, dtype=np.uint32)

    dem_texture = np.zeros((64, 64, 3), dtype=np.uint8)
    dem_texture[::2, :, 2] = 90
    dem_texture[:, ::2, 1] = 60
    dem_file = write_gltf(os.path.join(root, f'{flight_id}_matched_dem.gltf'),
                          vertices, indices, uvs, dem_texture)

    # ── Poses ────────────────────────────────────────────────────────────────
    fovy = fovy_covering(footprint_half, camera_height - dem_height)
    start_x = -step * (frame_count - 1) / 2.0
    positions = [(start_x + i * step, 0.0, camera_height) for i in range(frame_count)]

    poses = {'images': []}
    frame_files = []
    for idx, (x, y, z) in enumerate(positions):
        name = f'{flight_id}_{idx:08d}.png'
        poses['images'].append({
            'imagefile': name,
            'location': [float(x), float(y), float(z)],
            'rotation': [0.0, 0.0, 0.0],
            'fovy': [float(fovy)],
        })

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # A per-frame identifiable background plus a fixed structural pattern, so a render
        # can be attributed to the frame it came from.
        frame[..., 0] = int(30 + 40 * idx) % 256
        frame[height // 4:3 * height // 4, width // 4:3 * width // 4, 1] = 200
        frame[..., 2] = rng.integers(0, 40, size=(height, width), dtype=np.uint8)

        path = os.path.join(frames_dir, name)
        cv2.imwrite(path, frame)
        frame_files.append(path)

    poses_file = os.path.join(root, f'{flight_id}_matched_poses.json')
    with open(poses_file, 'w') as handle:
        json.dump(poses, handle)

    # ── Correction (neutral, so geometry stays analytically predictable) ─────
    correction_file = os.path.join(root, f'{flight_id}_correction.json')
    with open(correction_file, 'w') as handle:
        json.dump({
            'translation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'rotation': {'x': 0.0, 'y': 0.0, 'z': 0.0},
        }, handle)

    # ── Mask (fully opaque, so it cannot silently hide a projection bug) ─────
    mask_file = os.path.join(root, f'{flight_id}_mask_t.png')
    cv2.imwrite(mask_file, np.full((height, width, 3), 255, dtype=np.uint8))

    # ── Ground objects and MOT labels ───────────────────────────────────────
    if objects is None:
        objects = [(1, 0.0, 0.0), (2, 6.0, -3.0)]

    mot_lines = []
    for frame_idx, (cam_x, cam_y, _) in enumerate(positions):
        for track_id, world_x, world_y in objects:
            # Invert the pinhole projection of a camera looking straight down at a flat DEM.
            ndc_x = (world_x - cam_x) / footprint_half
            ndc_y = (world_y - cam_y) / footprint_half
            if abs(ndc_x) > 1.0 or abs(ndc_y) > 1.0:
                continue
            px = (ndc_x + 1.0) * 0.5 * width
            py = (1.0 - ndc_y) * 0.5 * height
            box = 6.0
            mot_lines.append(
                f'{frame_idx},{track_id},{px - box / 2:.3f},{py - box / 2:.3f},{box:.3f},{box:.3f},1,1,1'
            )

    mot_file = os.path.join(root, f'{flight_id}_gt.txt')
    with open(mot_file, 'w') as handle:
        handle.write('\n'.join(mot_lines) + ('\n' if mot_lines else ''))

    return SyntheticFlight(
        root=root,
        dem_file=dem_file,
        poses_file=poses_file,
        correction_file=correction_file,
        mask_file=mask_file,
        frame_files=frame_files,
        mot_file=mot_file,
        dem_height=dem_height,
        camera_height=camera_height,
        fovy=fovy,
        frame_size=frame_size,
        footprint_half=footprint_half,
        positions=positions,
        objects=list(objects),
    )
