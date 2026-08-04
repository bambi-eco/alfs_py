"""High-level wrappers for orthographic projection and airborne light field (ALFS) rendering.

This module abstracts the boilerplate shown in the BAMBI dataset introduction notebook
(loading the DEM mesh, poses, correction and mask, building the render context, creating
shots/cameras, rendering, and projecting bounding-box labels) into a single reusable
``ProjectionScene`` object with two methods:

* :meth:`ProjectionScene.project_orthographic` -- render the orthographic projection of a
  single frame (optionally projecting that frame's labels into the rendered image).
* :meth:`ProjectionScene.render_lightfield` -- render the integral (ALFS) projection from a
  central frame plus neighbouring frames (optionally projecting and merging labels from all
  contributing frames per track).

Typical usage::

    from alfspy.render.projection import ProjectionScene, ProjectionSettings, parse_mot_labels

    labels_by_frame = parse_mot_labels("interpolated_mot/146_gt.txt")

    with ProjectionScene(
        dem_file="bambi_downloads/146_matched_dem.glb",
        poses_file="bambi_downloads/146_matched_poses.json",
        correction_file="bambi_downloads/146_correction.json",
        mask_file="bambi_downloads/146_mask_t.png",
        settings=ProjectionSettings(ortho_size=(70, 70)),
    ) as scene:

        # Orthographic projection of a single frame
        scene.project_orthographic(
            "146_frames/thermal/146_00002120.png",
            labels=labels_by_frame.get(2120),
            output_image="146_ortho/146_00002120.png",
            output_labels="146_ortho/146_00002120.txt",
        )

        # ALFS / light field sample from a central frame and its neighbours
        scene.render_lightfield(
            central_image="146_frames/thermal/146_00002125.png",
            neighbor_images=[f"146_frames/thermal/146_000021{n:02d}.png"
                             for n in list(range(10, 25)) + list(range(26, 41))],
            labels=labels_by_frame,
            output_image="146_alfs/146_00002125.png",
            output_labels="146_alfs/146_00002125.txt",
        )
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from pyrr import Matrix44, Quaternion, Vector3
from trimesh import Trimesh

from alfspy.core.torchgl import TorchContext

from alfspy.core.convert.convert import pixel_to_world_coord, world_to_pixel_coord
from alfspy.core.geo.transform import Transform
from alfspy.core.rendering import (
    Camera,
    CtxShot,
    RenderResultMode,
    Resolution,
    TextureData,
)
from alfspy.core.rendering.renderer import Renderer
from alfspy.core.util.geo import get_aabb
from alfspy.core.util.pyrrs import quaternion_from_eulers
from alfspy.render.data import BaseSettings, CameraPositioningMode
from alfspy.render.render import (
    make_camera,
    make_shot_loader,
    make_torch_context,
    process_render_data,
    read_gltf,
    release_all,
)

# ───────────────────────────── Data structures ──────────────────────────────


@dataclass
class ProjectionSettings:
    """Render settings shared by both projection methods.

    :cvar ortho_size: Width/height of the orthographic frustum in world units (metres).
    :cvar render_resolution: Resolution of the rendered output image.
    :cvar fovy: Field-of-view used as fallback when a pose entry does not provide one.
    :cvar aspect_ratio: Aspect ratio of the shot cameras.
    :cvar add_background: Whether to overlay the result on a render of the background mesh.
    :cvar input_resolution: Resolution of the source frames. If ``None`` it is derived from the
        mask (when given) or from the first projected image.
    """
    ortho_size: Tuple[float, float] = (70.0, 70.0)
    render_resolution: Resolution = field(default_factory=lambda: Resolution(2048, 2048))
    fovy: float = 50.0
    aspect_ratio: float = 1.0
    add_background: bool = False
    input_resolution: Optional[Resolution] = None


@dataclass
class Label:
    """A 2D axis-aligned bounding box in the *source* image's pixel space.

    :cvar class_id: The object class id written to the YOLO label file.
    :cvar bbox: Bounding box as ``(x, y, width, height)`` in source-image pixels.
    :cvar track_id: Track id used to merge boxes across frames in :meth:`render_lightfield`.
        Required for light-field merging; ignored by :meth:`project_orthographic`.
    """
    class_id: int
    bbox: Tuple[float, float, float, float]
    track_id: Optional[int] = None

    def corners(self) -> List[float]:
        """:return: The four corners as a flat ``[x1, y1, x2, y1, x2, y2, x1, y2]`` list."""
        x, y, w, h = self.bbox
        x2, y2 = x + w, y + h
        return [x, y, x2, y, x2, y2, x, y2]


@dataclass
class ProjectedLabel:
    """A label projected into rendered-image pixel space.

    :cvar class_id: The object class id.
    :cvar bbox: Axis-aligned box as ``(x_min, y_min, x_max, y_max)`` in render pixels.
    :cvar track_id: The originating track id (if any).
    :cvar yolo: The YOLO formatted string, or ``None`` if the box lies fully outside the image.
    """
    class_id: int
    bbox: Tuple[float, float, float, float]
    track_id: Optional[int] = None
    yolo: Optional[str] = None


# ─────────────────────────────── Label helpers ──────────────────────────────


def parse_mot_labels(mot_file: str, class_id_column: int = 1) -> Dict[int, List[Label]]:
    """Parse a (comma separated) MOT ground-truth file into per-frame labels.

    The expected layout is ``frame_id, track_id, bb_left, bb_top, bb_width, bb_height, ...``.

    :param mot_file: Path to the MOT file.
    :param class_id_column: Column index to read the class id from. Defaults to ``1`` (the
        track id column) to match the BAMBI introduction notebook; pass ``7`` for the class
        column of a standard MOT17 file.
    :return: A dict mapping ``frame_id`` to the list of :class:`Label` objects in that frame.
    """
    labels_by_frame: Dict[int, List[Label]] = {}
    if not os.path.exists(mot_file):
        return labels_by_frame

    with open(mot_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = line.split(",")
            if len(values) < 6:
                continue
            frame_id = int(values[0])
            track_id = int(values[1])
            class_id = int(float(values[class_id_column])) if class_id_column < len(values) else track_id
            x, y, w, h = (float(v) for v in values[2:6])
            labels_by_frame.setdefault(frame_id, []).append(
                Label(class_id=class_id, bbox=(x, y, w, h), track_id=track_id)
            )
    return labels_by_frame


def _to_yolo(bbox: Tuple[float, float, float, float], img_width: int, img_height: int) -> Optional[str]:
    """Convert ``(x_min, y_min, x_max, y_max)`` render pixels to a normalised YOLO string.

    :return: ``"x_center y_center width height"`` or ``None`` if the box lies fully outside
        the rendered image.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    if (x_center + width / 2 < 0 or x_center - width / 2 > 1
            or y_center + height / 2 < 0 or y_center - height / 2 > 1):
        return None

    return f"{x_center} {y_center} {width} {height}"


# ─────────────────────────────── Scene object ───────────────────────────────


class ProjectionScene:
    """Loads a flight's DEM, poses, correction and mask once and renders projections from it.

    A single scene owns one render context, the processed mesh/texture and a ray-casting
    :class:`~trimesh.Trimesh`. Reuse it across many frames to avoid reloading the (potentially
    large) DEM for every render. Call :meth:`release` when finished, or use the scene as a
    context manager.
    """

    def __init__(
        self,
        dem_file: str,
        poses_file: str,
        correction_file: str,
        mask_file: Optional[str] = None,
        settings: Optional[ProjectionSettings] = None,
        ctx: Optional[TorchContext] = None,
        device: Optional[str] = None,
        gl_backend: Optional[str] = None,
    ):
        """
        :param dem_file: Path to the DEM mesh (``.glb``/``.gltf``).
        :param poses_file: Path to the matched poses JSON (``..._matched_poses.json``).
        :param correction_file: Path to the flight correction JSON (``..._correction.json``).
        :param mask_file: Optional path to a projection mask image.
        :param settings: Render settings. Defaults to :class:`ProjectionSettings` defaults.
        :param ctx: An existing :class:`~alfspy.core.torchgl.context.TorchContext` to render
            in. If ``None`` one is created (and released by :meth:`release`).
        :param device: The torch device to render on, e.g. ``"cpu"`` or ``"cuda"`` (optional).
            Used only when ``ctx`` is ``None``; defaults to CUDA when available.
        :param gl_backend: Deprecated and ignored. The torch backend needs no GL driver, so
            there is no headless backend to select -- this is the parameter the Xvfb-based
            deployment used to need.
        """
        self.settings = settings or ProjectionSettings()

        # ── Poses ────────────────────────────────────────────────────────────
        with open(poses_file, "r") as f:
            self.poses = json.load(f)
        self.images = self.poses["images"]

        # ── Correction (transform + raw euler/translation for label cameras) ─
        self.correction, self._cor_translation, self._cor_rotation_eulers = _load_correction(correction_file)

        # ── Render context ───────────────────────────────────────────────────
        if ctx is None:
            self.ctx = make_torch_context(device=device)
            self._owns_ctx = True
        else:
            self.ctx = ctx
            self._owns_ctx = False

        # ── Mask ─────────────────────────────────────────────────────────────
        self.mask_texture: Optional[TextureData] = None
        if mask_file and os.path.exists(mask_file):
            mask_img = CtxShot._cvt_img(cv2.imread(mask_file, cv2.IMREAD_UNCHANGED))
            self.mask_texture = TextureData(mask_img)
            if self.settings.input_resolution is None:
                self.settings.input_resolution = Resolution(mask_img.shape[1], mask_img.shape[0])

        # ── Mesh / texture / ray-casting mesh ────────────────────────────────
        mesh_data, texture_data = read_gltf(dem_file)
        # Build the ray-casting mesh from the raw vertices before process_render_data alters them.
        self.tri_mesh = Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)
        self.mesh_data, self.texture_data = process_render_data(mesh_data, texture_data)
        self.mesh_aabb = get_aabb(self.mesh_data.vertices)

        self._released = False

    # region context manager / cleanup

    def __enter__(self) -> "ProjectionScene":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    def release(self) -> None:
        """Release the mesh/texture and (if owned) the render context."""
        if self._released:
            return
        self.mesh_data = None
        self.texture_data = None
        if self._owns_ctx:
            release_all(self.ctx)
        self._released = True

    # endregion

    # region internal helpers

    def _build_settings(self) -> BaseSettings:
        return BaseSettings(
            count=len(self.images),
            initial_skip=0,
            add_background=self.settings.add_background,
            camera_position_mode=CameraPositioningMode.FirstShot,
            fovy=self.settings.fovy,
            aspect_ratio=self.settings.aspect_ratio,
            orthogonal=True,
            ortho_size=tuple(self.settings.ortho_size),
            correction=self.correction,
            resolution=self.settings.render_resolution,
        )

    @staticmethod
    def _rotation_for(meta: dict) -> Quaternion:
        """
        Builds the shot rotation from a pose entry.

        Shared by :meth:`_make_shot` and :meth:`_camera_for_frame` so the camera used to
        *render* a frame and the camera used to *un-project its labels* cannot drift apart.

        :param meta: The pose entry for one frame.
        :return: The rotation of that frame's camera.
        """
        rotation = [val % 360.0 for val in meta["rotation"]]
        if len(rotation) == 3:
            return quaternion_from_eulers([np.deg2rad(v) for v in rotation], "zyx")
        if len(rotation) == 4:
            return Quaternion(rotation)
        raise ValueError(f"Invalid rotation format of length {len(rotation)}: {rotation}")

    def _make_shot(self, image_path: str, frame_idx: int) -> CtxShot:
        """Create a :class:`CtxShot` for ``image_path`` from the pose at ``frame_idx``."""
        meta = self.images[frame_idx]
        position = Vector3(meta["location"])
        rotation = self._rotation_for(meta)

        fov = _get_fovy(meta, self.settings.fovy)
        return CtxShot(self.ctx, image_path, position, rotation, fov, 1, self.correction, lazy=False)

    def _input_resolution(self, sample_image: str) -> Resolution:
        if self.settings.input_resolution is not None:
            return self.settings.input_resolution
        img = cv2.imread(sample_image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not read image to infer input resolution: {sample_image}")
        res = Resolution(img.shape[1], img.shape[0])
        self.settings.input_resolution = res
        return res

    def _project_label(self, label: Label, frame_idx: int, input_resolution: Resolution,
                       single_shot_camera: Camera) -> Tuple[float, float, float, float]:
        """Project one source-image label into render pixel space, returning its AABB."""
        camera = self._camera_for_frame(frame_idx)
        coords = label.corners()
        pixel_xs = [int(float(v)) for v in coords[0::2]]
        pixel_ys = [int(float(v)) for v in coords[1::2]]

        world = pixel_to_world_coord(
            pixel_xs, pixel_ys, input_resolution.width, input_resolution.height,
            self.tri_mesh, camera, include_misses=False,
        )
        xs, ys = world_to_pixel_coord(
            world, self.settings.render_resolution.width, self.settings.render_resolution.height,
            single_shot_camera,
        )
        xs = np.atleast_1d(np.asarray(xs))
        ys = np.atleast_1d(np.asarray(ys))
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def _camera_for_frame(self, frame_idx: int) -> Camera:
        """
        Camera matching the original source frame, used to ray-cast labels onto the mesh.

        This is derived from the *same* matrices the renderer uses for that frame's shot --
        the shot camera's view matrix composed with the correction, exactly as the shot
        program applies them (``P * C * V``). Label un-projection and image projection
        therefore agree by construction.

        Previously this rebuilt the camera by hand as ``position + correction_translation``
        and ``-(eulers - correction_eulers)``. The negation existed to cancel a transposed
        rotation in ``pixel_to_world_coord``; both have been fixed. The negation was also
        only *approximately* a transpose: ``R(-e) == R(e).T`` holds for a single-axis
        rotation but not for composed Euler angles, so any frame with real pitch or roll had
        its labels projected through a camera off by several degrees. Near-nadir flights hid
        this because their poses are effectively yaw-only.

        :param frame_idx: Index into the poses ``images`` list.
        :return: A camera whose view matrix equals the renderer's corrected shot view.
        """
        meta = self.images[frame_idx]
        fovy = _get_fovy(meta, self.settings.fovy)

        shot_camera = Camera(fovy=fovy, aspect_ratio=self.settings.aspect_ratio,
                             position=Vector3(meta["location"]),
                             rotation=self._rotation_for(meta))

        correction = Transform(self.correction.position, self.correction.rotation,
                               self.correction.scale, dtype="f4")
        correction_matrix = correction.trans_mat.inverse * correction.rot_mat

        view = (np.asarray(shot_camera.get_view(), dtype=np.float64)
                @ np.asarray(correction_matrix, dtype=np.float64))

        return _camera_from_view(view, fovy, self.settings.aspect_ratio)

    # endregion

    # region public API

    def project_orthographic(
        self,
        image_path: str,
        frame_idx: Optional[int] = None,
        labels: Optional[Sequence[Label]] = None,
        output_image: Optional[str] = None,
        output_labels: Optional[str] = None,
    ) -> List[ProjectedLabel]:
        """Render the orthographic projection of a single frame.

        :param image_path: Path to the source frame image.
        :param frame_idx: Index into the poses ``images`` list. If ``None`` it is parsed from
            the file name (``<flight>_<index>.<ext>``).
        :param labels: Optional labels of this frame to project into the rendered image.
        :param output_image: If given, the rendered image is saved to this path.
        :param output_labels: If given, projected labels are written here in YOLO format.
        :return: The projected labels (in render pixel space).
        """
        self._ensure_active()
        if frame_idx is None:
            frame_idx = frame_index_from_path(image_path)

        settings = self._build_settings()
        shot = self._make_shot(image_path, frame_idx)
        renderer = None
        try:
            single_shot_camera = make_camera(
                self.mesh_aabb, [shot], settings,
                rotation=Quaternion.from_matrix(shot.get_view().inverse @ shot.get_correction().inverse),
            )
            renderer = Renderer(settings.resolution, self.ctx, single_shot_camera,
                                self.mesh_data, self.texture_data)

            if output_image:
                _ensure_parent(output_image)
                renderer.project_shots(
                    make_shot_loader([shot]),
                    RenderResultMode.ShotOnly,
                    mask=self.mask_texture,
                    integral=False,
                    save=True,
                    release_shots=False,
                    save_name_iter=iter([output_image]),
                )

            projected: List[ProjectedLabel] = []
            if labels:
                input_resolution = self._input_resolution(image_path)
                for label in labels:
                    bbox = self._project_label(label, frame_idx, input_resolution, single_shot_camera)
                    yolo = _to_yolo(bbox, settings.resolution.width, settings.resolution.height)
                    projected.append(ProjectedLabel(label.class_id, bbox, label.track_id, yolo))

            if output_labels is not None:
                _write_yolo(output_labels, projected)

            return projected
        finally:
            release_all(renderer, shot)

    def render_lightfield(
        self,
        central_image: str,
        neighbor_images: Sequence[str],
        central_idx: Optional[int] = None,
        neighbor_indices: Optional[Sequence[int]] = None,
        labels: Optional[Dict[int, Sequence[Label]]] = None,
        output_image: Optional[str] = None,
        output_labels: Optional[str] = None,
        alpha_threshold: float = 0.1,
    ) -> List[ProjectedLabel]:
        """Render the integral (ALFS) projection from a central frame and its neighbours.

        The camera is anchored on the central frame; neighbour frames are integrated to
        synthesise the light-field sample. Labels from every contributing frame are projected
        into the central camera and merged per ``track_id`` into a single enclosing box.

        :param central_image: Path to the central frame image.
        :param neighbor_images: Paths to the neighbouring frame images (before and after).
        :param central_idx: Pose index of the central frame. Parsed from the file name if ``None``.
        :param neighbor_indices: Pose indices of the neighbours. Parsed from file names if ``None``.
        :param labels: Optional mapping ``frame_idx -> labels`` for all contributing frames.
            Labels must carry a ``track_id`` to be merged across frames.
        :param output_image: If given, the rendered integral image is saved to this path.
        :param output_labels: If given, merged labels are written here in YOLO format.
        :param alpha_threshold: Alpha threshold passed to the integral renderer.
        :return: The merged projected labels (one per track, in render pixel space).
        """
        self._ensure_active()
        if central_idx is None:
            central_idx = frame_index_from_path(central_image)
        if neighbor_indices is None:
            neighbor_indices = [frame_index_from_path(p) for p in neighbor_images]
        if len(neighbor_indices) != len(neighbor_images):
            raise ValueError("neighbor_indices must match neighbor_images in length")

        settings = self._build_settings()
        central_shot = self._make_shot(central_image, central_idx)
        neighbor_shots = [self._make_shot(p, idx) for p, idx in zip(neighbor_images, neighbor_indices)]
        renderer = None
        try:
            single_shot_camera = make_camera(
                self.mesh_aabb, [central_shot], settings,
                rotation=Quaternion.from_matrix(
                    central_shot.get_view().inverse @ central_shot.get_correction().inverse),
            )
            renderer = Renderer(settings.resolution, self.ctx, single_shot_camera,
                                self.mesh_data, self.texture_data)

            if output_image:
                _ensure_parent(output_image)
                all_shots = neighbor_shots[: len(neighbor_shots) // 2] + [central_shot] \
                    + neighbor_shots[len(neighbor_shots) // 2:]
                renderer.render_integral(
                    make_shot_loader(all_shots),
                    mask=self.mask_texture,
                    save=True,
                    release_shots=False,
                    save_name=output_image,
                    alpha_threshold=alpha_threshold,
                )

            merged: List[ProjectedLabel] = []
            if labels:
                input_resolution = self._input_resolution(central_image)
                all_indices = list(neighbor_indices) + [central_idx]

                # track_id -> list of (class_id, aabb) projected from every contributing frame
                per_track: Dict[object, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
                for fid in all_indices:
                    for li, label in enumerate(labels.get(fid, []) or []):
                        bbox = self._project_label(label, fid, input_resolution, single_shot_camera)
                        key = label.track_id if label.track_id is not None else f"{fid}:{li}"
                        per_track.setdefault(key, []).append((label.class_id, bbox))

                for track_key, states in per_track.items():
                    class_id = states[0][0]
                    boxes = np.array([s[1] for s in states])  # (n, 4) of (x_min,y_min,x_max,y_max)
                    x_min = float(boxes[:, 0].min())
                    y_min = float(boxes[:, 1].min())
                    x_max = float(boxes[:, 2].max())
                    y_max = float(boxes[:, 3].max())
                    enclosing = (x_min, y_min, x_max, y_max)
                    yolo = _to_yolo(enclosing, settings.resolution.width, settings.resolution.height)
                    track_id = track_key if isinstance(track_key, int) else None
                    merged.append(ProjectedLabel(class_id, enclosing, track_id, yolo))

            if output_labels is not None:
                _write_yolo(output_labels, merged)

            return merged
        finally:
            release_all(renderer, central_shot, neighbor_shots)

    # endregion

    def _ensure_active(self) -> None:
        if self._released:
            raise RuntimeError("ProjectionScene has been released")


# ─────────────────────────────── Module helpers ─────────────────────────────


def frame_index_from_path(image_path: str) -> int:
    """Extract the frame index from a ``<flight>_<index>.<ext>`` file name."""
    return int(Path(image_path).stem.rsplit("_", 1)[1])


def _camera_from_view(view: np.ndarray, fovy: float, aspect_ratio: float) -> Camera:
    """
    Reconstructs a :class:`Camera` whose ``get_view()`` reproduces the given view matrix.

    Matrices are in pyrr's row-vector convention, so the inverse of a view matrix is the
    camera-to-world transform: its translation lives in row 3 and its rotation in the
    upper-left 3x3 block.

    :param view: A ``(4, 4)`` world-to-camera matrix.
    :param fovy: The field of view to give the reconstructed camera.
    :param aspect_ratio: The aspect ratio to give the reconstructed camera.
    :return: A camera equivalent to the given view.
    """
    to_world = np.linalg.inv(np.asarray(view, dtype=np.float64))

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = to_world[:3, :3]

    return Camera(
        fovy=fovy,
        aspect_ratio=aspect_ratio,
        position=Vector3(to_world[3, :3].astype("f4")),
        rotation=Quaternion.from_matrix(Matrix44(rotation_matrix.astype("f4"))),
    )


def _get_fovy(meta: dict, fallback: float) -> float:
    fov = meta.get("fovy", fallback)
    if fov is None:
        return float(fallback)
    if isinstance(fov, (list, tuple)):
        return float(fov[0]) if len(fov) else float(fallback)
    return float(fov)


def _load_correction(correction_file: str) -> Tuple[Transform, Vector3, Vector3]:
    """:return: ``(transform, translation_vec, rotation_eulers_vec)`` from a correction JSON."""
    with open(correction_file, "r") as f:
        data = json.load(f)
    t = data.get("translation", {"x": 0.0, "y": 0.0, "z": 0.0})
    r = data.get("rotation", {"x": 0.0, "y": 0.0, "z": 0.0})
    translation = Vector3([t["x"], t["y"], t["z"]], dtype="f4")
    eulers = Vector3([r["x"], r["y"], r["z"]], dtype="f4")
    return Transform(translation, Quaternion.from_eulers(eulers)), translation, eulers


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_yolo(output_labels: str, projected: Sequence[ProjectedLabel]) -> int:
    _ensure_parent(output_labels)
    written = 0
    with open(output_labels, "w") as f:
        for label in projected:
            if label.yolo is not None:
                f.write(f"{label.class_id} {label.yolo}\n")
                written += 1
    return written
