import bpy
import os
import json
import struct
import numpy as np
import math
import sys
import importlib
import traceback
from pathlib import Path
import inspect
from mathutils import Quaternion, Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view
import gpu

# ------------------------------------------------------------------------
# Optional GLSL renderer import helper
# ------------------------------------------------------------------------

LCCGLSLRenderer = None
import_error_msg = ""

def _resolve_script_dir():
    """
    このスクリプトが置かれているディレクトリを返す。
    Text ブロックから実行した場合でもローカルモジュールを import できるようにするため。
    """
    # 通常の __file__ 経由
    module_file = getattr(sys.modules.get(__name__), "__file__", None)
    if module_file:
        path = Path(module_file).resolve()
        if path.exists():
            return path.parent

    # Text エディタから実行されている場合
    text_block = bpy.data.texts.get("LCC_Render.py")
    if text_block and text_block.filepath:
        text_path = Path(text_block.filepath).resolve()
        if text_path.exists():
            return text_path.parent

    # フレーム情報からのフォールバック
    try:
        frame_file = inspect.getsourcefile(sys.modules[__name__])
        if frame_file:
            path = Path(frame_file).resolve()
            if path.exists():
                return path.parent
    except Exception:
        pass

    return None

# スクリプトディレクトリを sys.path に追加して GLSL レンダラーを探す
_script_dir = _resolve_script_dir()
if _script_dir and str(_script_dir) not in sys.path:
    sys.path.append(str(_script_dir))

try:
    from LCC_GLSL_Renderer import LCCGLSLRenderer
except Exception as e:
    import_error_msg = str(e)
    print(f"ERROR: Failed to import LCC_GLSL_Renderer: {e}")
    traceback.print_exc()
    LCCGLSLRenderer = None

# ------------------------------------------------------------------------
# Globals for GLSL Renderer
# ------------------------------------------------------------------------
glsl_renderer = None
draw_handler = None
last_cam_pos = None # For sort optimization

def update_view(scene, context):
    # Callback to update sorting if needed
    pass

# ------------------------------------------------------------------------
# LCC Decoding Logic
# ------------------------------------------------------------------------

def decode_rotation(encoded_array):
    """
    Decodes the custom packed uint32 rotation format from LCC.
    Returns Quaternion (x, y, z, w).
    """
    sqrt2 = 1.414213562373095
    rsqrt2 = 0.7071067811865475
    QLut = np.array([3, 0, 1, 2, 0, 3, 1, 2, 0, 1, 3, 2, 0, 1, 2, 3], dtype=np.int32)

    v0 = (encoded_array & 1023) / 1023.0
    v1 = ((encoded_array >> 10) & 1023) / 1023.0
    v2 = ((encoded_array >> 20) & 1023) / 1023.0
    v3 = ((encoded_array >> 30) & 3) / 3.0

    pq = np.stack([v0, v1, v2, v3], axis=-1)

    idx = np.round(pq[:, 3] * 3.0).astype(np.int32)
    q_xyz = pq[:, :3] * sqrt2 - rsqrt2
    dot_prod = np.sum(q_xyz * q_xyz, axis=1)
    dot_prod = np.clip(dot_prod, 0.0, 1.0)
    q_w = np.sqrt(1.0 - dot_prod)
    
    q_vals = np.zeros((len(encoded_array), 4), dtype=np.float32)
    q_vals[:, 0] = q_xyz[:, 0]
    q_vals[:, 1] = q_xyz[:, 1]
    q_vals[:, 2] = q_xyz[:, 2]
    q_vals[:, 3] = q_w

    result = np.zeros((len(encoded_array), 4), dtype=np.float32)
    
    for i in range(4):
        lut_indices = idx * 4 + i
        component_indices = QLut[lut_indices]
        result[:, i] = q_vals[np.arange(len(encoded_array)), component_indices]
        
    return result

def quaternion_to_euler_numpy(quats):
    """
    Convert Quaternion (x, y, z, w) to Euler angles (XYZ) using pure numpy.
    """
    x = quats[:, 0]
    y = quats[:, 1]
    z = quats[:, 2]
    w = quats[:, 3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return np.stack((roll_x, pitch_y, yaw_z), axis=-1).astype(np.float32)

def make_lookat(eye, target, up):
    """
    OpenGL-style LookAt matrix built with mathutils.Matrix.
    Camera forward is assumed to be -Z.
    """
    f = (target - eye).normalized()
    s = f.cross(up).normalized()
    u = s.cross(f)

    m = Matrix((
        ( s.x,  u.x, -f.x, 0.0),
        ( s.y,  u.y, -f.y, 0.0),
        ( s.z,  u.z, -f.z, 0.0),
        (-s.dot(eye), -u.dot(eye),  f.dot(eye), 1.0),
    ))
    return m


def make_perspective(fov_y_rad, aspect, z_near, z_far):
    """
    OpenGL-style perspective projection matrix.
    fov_y_rad: vertical FOV in radians.
    """
    f = 1.0 / math.tan(fov_y_rad * 0.5)
    range_inv = 1.0 / (z_near - z_far)

    return Matrix((
        (f / aspect, 0.0,                        0.0,                           0.0),
        (0.0,        f,                          0.0,                           0.0),
        (0.0,        0.0, (z_far + z_near) * range_inv,  2.0 * z_far * z_near * range_inv),
        (0.0,        0.0,                       -1.0,                           0.0),
    ))


def cubemap_to_equirect(faces, width, height):
    """
    faces: dict[str, np.ndarray] (PX, NX, PY, NY, PZ, NZ) each (N, N, 4) uint8
    width, height: output resolution of the equirectangular image
    returns: (height, width, 4) uint8
    """
    out = np.zeros((height, width, 4), dtype=np.uint8)
    face_size = list(faces.values())[0].shape[0]

    # equirect grid
    u = (np.arange(width) + 0.5) / width
    v = (np.arange(height) + 0.5) / height
    uu, vv = np.meshgrid(u, v)

    phi = uu * 2.0 * np.pi - np.pi        # longitude: -pi .. pi
    theta = vv * np.pi - 0.5 * np.pi      # latitude: -pi/2 .. pi/2

    x = np.cos(theta) * np.cos(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.sin(phi)

    abs_x = np.abs(x)
    abs_y = np.abs(y)
    abs_z = np.abs(z)

    max_axis = np.argmax(
        np.stack([abs_x, abs_y, abs_z], axis=-1),
        axis=-1
    )

    # Helpers to sample from each cube face
    def sample_face(mask, sc, tc, face_key):
        if not np.any(mask):
            return
        sc = sc[mask]
        tc = tc[mask]
        u_face = (sc + 1.0) * 0.5
        v_face = (tc + 1.0) * 0.5
        ix = np.clip((u_face * (face_size - 1)).astype(np.int32), 0, face_size - 1)
        iy = np.clip((v_face * (face_size - 1)).astype(np.int32), 0, face_size - 1)
        out[mask] = faces[face_key][iy, ix]

    # +X
    mask = (max_axis == 0) & (x > 0)
    sample_face(mask, -z / abs_x,  y / abs_x, 'PX')

    # -X
    mask = (max_axis == 0) & (x < 0)
    sample_face(mask,  z / abs_x,  y / abs_x, 'NX')

    # +Y
    mask = (max_axis == 1) & (y > 0)
    sample_face(mask,  x / abs_y, -z / abs_y, 'PY')

    # -Y
    mask = (max_axis == 1) & (y < 0)
    sample_face(mask,  x / abs_y,  z / abs_y, 'NY')

    # +Z
    mask = (max_axis == 2) & (z > 0)
    sample_face(mask,  x / abs_z,  y / abs_z, 'PZ')

    # -Z
    mask = (max_axis == 2) & (z < 0)
    sample_face(mask, -x / abs_z,  y / abs_z, 'NZ')

    return out

class LCCParser:
    def __init__(self, meta_path):
        self.meta_path = meta_path
        self.dir_path = os.path.dirname(meta_path)
        self.meta = {}
        self.load_meta()

        # Meta.lcc transform terms (kept in float64 to avoid precision loss at large scales)
        self.meta_offset = np.array(self.meta.get("offset", [0, 0, 0]), dtype=np.float64)
        self.meta_shift = np.array(self.meta.get("shift", [0, 0, 0]), dtype=np.float64)
        self.meta_scale = np.array(self.meta.get("scale", [1, 1, 1]), dtype=np.float64)

    def load_meta(self):
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
            
    def get_lod_info(self, lod_level=0):
        index_path = os.path.join(self.dir_path, "Index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError("Index.bin not found")

        total_level = self.meta.get("totalLevel", 1)
        index_data_size = self.meta.get("indexDataSize", 0)

        # Respect indexDataSize when provided to stay spec-compliant
        if index_data_size and index_data_size > 0:
            unit_entry_size = index_data_size
        else:
            unit_entry_size = 4 + (4 + 8 + 4) * total_level
        
        units = []
        file_size = os.path.getsize(index_path)
        num_units = file_size // unit_entry_size
        
        with open(index_path, 'rb') as f:
            for i in range(num_units):
                data = f.read(unit_entry_size)
                if len(data) < 4 + (lod_level + 1) * 16:
                    # Not enough bytes for this LOD entry in this unit
                    continue
                base_off = 4 + lod_level * 16
                
                count = struct.unpack_from('<I', data, base_off)[0]
                offset = struct.unpack_from('<Q', data, base_off + 4)[0]
                size = struct.unpack_from('<I', data, base_off + 12)[0]
                
                if count > 0:
                    units.append({
                        'count': count,
                        'offset': offset,
                        'size': size
                    })
        return units

    def iter_chunks(self, units, batch_size=1000000):
        data_path = os.path.join(self.dir_path, "Data.bin")
        if not os.path.exists(data_path):
            raise FileNotFoundError("Data.bin not found")
            
        current_pos = []
        current_col = []
        current_scl = []
        current_rot = []
        current_extra = []
        current_count = 0
        
        with open(data_path, 'rb') as f:
            for u in units:
                f.seek(u['offset'])
                raw_bytes = f.read(u['size'])
                count = u['count']
                
                if len(raw_bytes) < count * 32:
                    print(f"Warning: Unit data smaller than expected. Skipping.")
                    continue
                
                chunk_data = np.frombuffer(raw_bytes[:count*32], dtype=np.uint8)
                chunk_matrix = chunk_data.reshape((count, 32))
                
                pos_raw = chunk_matrix[:, 0:12].copy()
                pos_data = pos_raw.view(dtype=np.float32).reshape(count, 3)
                
                col_raw = chunk_matrix[:, 12:16].copy()
                col_data = col_raw.view(dtype=np.uint32).reshape(count)
                
                scl_raw = chunk_matrix[:, 16:22].copy()
                scl_data = scl_raw.view(dtype=np.uint16).reshape(count, 3)
                
                rot_raw = chunk_matrix[:, 22:26].copy()
                rot_data = rot_raw.view(dtype=np.uint32).reshape(count)

                extra_raw = chunk_matrix[:, 26:32].copy()  # reserved/opacity/etc.
                
                current_pos.append(pos_data)
                current_col.append(col_data)
                current_scl.append(scl_data)
                current_rot.append(rot_data)
                current_extra.append(extra_raw)
                current_count += count
                
                if current_count >= batch_size:
                    yield (
                        self._apply_meta_transform(np.concatenate(current_pos)),
                        np.concatenate(current_col),
                        np.concatenate(current_scl),
                        np.concatenate(current_rot),
                        np.concatenate(current_extra)
                    )
                    current_pos = []
                    current_col = []
                    current_scl = []
                    current_rot = []
                    current_extra = []
                    current_count = 0
                    
        if current_count > 0:
            yield (
                self._apply_meta_transform(np.concatenate(current_pos)),
                np.concatenate(current_col),
                np.concatenate(current_scl),
                np.concatenate(current_rot),
                np.concatenate(current_extra)
            )

    def _apply_meta_transform(self, positions):
        """Apply Meta.lcc offset/shift/scale to positions using float64 to avoid precision loss."""
        pos64 = np.asarray(positions, dtype=np.float64)
        return (pos64 + self.meta_offset + self.meta_shift) * self.meta_scale

    def decode_attributes(self, colors_u, scales_u, rots_u, extra_u=None, apply_exp_scale=False):
        """Decode packed attributes. Opacity is applied only when non-zero data exists."""

        # RGBA8 -> float (0..1)
        colors = colors_u.view(dtype=np.uint8).reshape(-1, 4).astype(np.float32) / 255.0

        # Scale decode
        scale_meta = next((a for a in self.meta.get('attributes', []) if a['name'] == 'scale'), None)
        if scale_meta:
            s_min = np.array(scale_meta['min'], dtype=np.float32)
            s_max = np.array(scale_meta['max'], dtype=np.float32)
        else:
            s_min = np.array([0, 0, 0], dtype=np.float32)
            s_max = np.array([1, 1, 1], dtype=np.float32)

        scales_norm = scales_u.astype(np.float32) / 65535.0
        scales = s_min + scales_norm * (s_max - s_min)

        # Optional: revert log scale (3DGS export) only when requested
        if apply_exp_scale:
            scales = np.exp(scales)

        # Rotation decode
        quats_np = decode_rotation(rots_u)
        eulers = quaternion_to_euler_numpy(quats_np)

        # Optional opacity from reserved bytes (first uint16 slot)
        opacity = None
        if extra_u is not None and len(extra_u) > 0:
            extra_u16 = extra_u.view(dtype=np.uint16).reshape(-1, 3)
            op_raw = extra_u16[:, 0]

            # Ignore if entirely zero (means opacity is unused in the source)
            if np.any(op_raw):
                opacity_meta = next(
                    (a for a in self.meta.get('attributes', []) if a['name'] == 'opacity'),
                    None
                )

                def _scalar(v, default):
                    if isinstance(v, (list, tuple, np.ndarray)):
                        return float(v[0]) if len(v) > 0 else float(default)
                    try:
                        return float(v)
                    except Exception:
                        return float(default)

                if opacity_meta:
                    op_min = _scalar(opacity_meta.get('min', 0.0), 0.0)
                    op_max = _scalar(opacity_meta.get('max', 1.0), 1.0)
                else:
                    op_min, op_max = 0.0, 1.0

                if op_max != op_min:
                    opacity = op_min + (op_raw.astype(np.float32) / 65535.0) * (op_max - op_min)
                else:
                    opacity = np.full_like(op_raw, op_min, dtype=np.float32)

                # Apply alpha modulation only when opacity is present
                colors[:, 3] *= opacity

        # Safety: if alpha collapsed to (near) zero everywhere, fall back to fully opaque
        if np.all(colors[:, 3] <= 1e-3):
            colors[:, 3] = 1.0
            opacity = None

        return colors, scales, quats_np, eulers, opacity


class LCC_DisplaySettings(bpy.types.PropertyGroup):
    # Geometry Nodes側
    scale_density: bpy.props.FloatProperty(
        name="スケール倍率",
        default=1.5,
        min=0.01,
    )
    min_thickness: bpy.props.FloatProperty(
        name="最小厚み",
        default=0.05,
        min=0.0,
    )
    lod_distance: bpy.props.FloatProperty(
        name="LOD距離",
        default=100.0,
        min=0.0,
    )
    scale_filter_max: bpy.props.FloatProperty(
        name="スケール上限フィルタ",
        default=5.0,
        min=0.0,
    )

    # マテリアル側（ガウスの形と濃さ）
    gauss_k: bpy.props.FloatProperty(
        name="ガウス係数 k",
        default=1.0,
        min=0.01,
    )
    alpha_threshold: bpy.props.FloatProperty(
        name="アルファしきい値",
        default=0.001,
        min=0.0,
    )
    alpha_boost: bpy.props.FloatProperty(
        name="アルファ補正",
        default=1.0,
        min=0.0,
    )

# ------------------------------------------------------------------------
# Blender Operator
# ------------------------------------------------------------------------

class IMPORT_OT_lcc(bpy.types.Operator):
    bl_idname = "import_scene.lcc"
    bl_label = "LCCをインポート（ビュー/レンダー分割）"
    bl_description = (
        "LCCデータ（XGRIDS発のデータオーガナイズ形式）を読み込みます。"
        "仕様: https://github.com/xgrids/LCCWhitepaper"
    )
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    lod_min: bpy.props.IntProperty(name="最小LOD", default=0, min=0, max=5)
    lod_max: bpy.props.IntProperty(name="最大LOD", default=0, min=0, max=8)

    scale_density: bpy.props.FloatProperty(name="スケール倍率", default=1.0, min=0.1)
    min_thickness: bpy.props.FloatProperty(name="最小厚み", default=0.05, min=0.0)

    chunk_size: bpy.props.FloatProperty(name="チャンクグリッドサイズ", default=10.0, min=1.0, description="レンダーチャンクを分割するグリッドセルの大きさ")

    lod_distance: bpy.props.FloatProperty(name="描画距離", default=100.0, min=1.0, description="この距離以内のポイントをフル描画します")

    culling_distance: bpy.props.FloatProperty(name="チャンクカリング距離", default=0.0, min=0.0, description="この距離より遠いチャンクをレンダーで無効化します (0 で無効)")

    # Determines how much data is sampled for the viewport proxy object
    viewport_density: bpy.props.FloatProperty(
        name="ビューポート密度(%)",
        default=1.0,
        min=0.1,
        max=100.0,
        description="GLSL ビューポートレンダラーに使うポイントの割合"
    )

    setup_render: bpy.props.BoolProperty(name="360°レンダーを設定", default=True)
    show_bounds: bpy.props.BoolProperty(name="バウンディングを表示", default=True)

    use_glsl: bpy.props.BoolProperty(name="GLSLビューポートを使用", default=True, description="ビューポートで高速な OpenGL レンダラーを使います")

    # 3DGS compatibility
    use_3dgs_attributes: bpy.props.BoolProperty(
        name="3DGS属性を使用",
        default=False,
        description="3DGS形式のLCCを正しく復号するためにログスケール等を展開します（チャンクとLCC_GN系ビューアは常に自前で生成）"
    )

    g3ds_alpha_boost: bpy.props.FloatProperty(
        name="3DGSアルファ補正",
        default=2.0,
        min=0.1,
        max=5.0,
        description="3DGSモード時にだけアルファ(不透明度)を持ち上げて濃く見せる係数"
    )

    g3ds_scale_boost: bpy.props.FloatProperty(
        name="3DGSスケール補正",
        default=1.0,
        min=0.1,
        max=3.0,
        description="3DGSモード時にだけスプラットのスケールを補正する係数"
    )

    g3ds_node_group: bpy.props.StringProperty(
        name="3DGSノードグループ",
        default="Gaussian splatting",
        description="3DGS PLY属性を消費する既存の Geometry Nodes グループ名"
    )
    rot_wxyz: bpy.props.BoolProperty(
        name="回転をWXYZ順で保存",
        default=False,
        description="有効にすると rot_0..3 を x,y,z,w ではなく w,x,y,z の順で保存します"
    )

    def execute(self, context):
        if not self.filepath.lower().endswith(".lcc"):
            self.report({'ERROR'}, ".lcc ファイルを選択してください。")
            return {'CANCELLED'}
            
        self.cleanup_old_objects()
        
        # Collections setup
        render_coll_name = "LCC_Render"
        
        if render_coll_name not in bpy.data.collections:
            render_coll = bpy.data.collections.new(render_coll_name)
            context.scene.collection.children.link(render_coll)
        else:
            render_coll = bpy.data.collections[render_coll_name]

        # --- インポート時のパラメータをシーン設定に保存 ---
        settings = getattr(context.scene, "lcc_display_settings", None)
        if settings is not None:
            settings.scale_density = self.scale_density
            settings.min_thickness = self.min_thickness
            settings.lod_distance = self.lod_distance
            
        # Camera setup
        if self.setup_render:
            self.setup_360_camera()
            self.optimize_cycles_for_transparency()
            if context.scene.camera:
                context.scene.camera.data.clip_end = 10000.0
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        for space in area.spaces:
                            if space.type == 'VIEW_3D':
                                space.clip_end = 10000.0

        parser = LCCParser(self.filepath)
        
        target_lod_start = self.lod_min
        target_lod_end = self.lod_max
        
        start_lod = min(target_lod_start, target_lod_end)
        end_lod = max(target_lod_start, target_lod_end)
        
        total_points = 0
        
        # Buffer for Viewport Object (Unified)
        vp_pos_list, vp_col_list, vp_scl_list, vp_rot_list, vp_rot_quat_list, vp_opa_list = [], [], [], [], [], []
        
        # Spatial Grid Buffer for Render Chunks
        # Key: (grid_x, grid_y, grid_z), Value: lists of (pos, col, scl, rot, lod)
        spatial_chunks = {}
        chunk_grid_size = self.chunk_size

        try:
            for lvl in range(start_lod, end_lod + 1):
                units = parser.get_lod_info(lvl)
                if not units: continue
                
                for pos, col_u, scl_u, rot_u, extra_u in parser.iter_chunks(units, batch_size=1000000):
                    # Decode
                    # Note: decode_attributes now returns quats AND eulers
                    # apply_exp_scale=False keeps LCC default scale; set True only if the source was log-scaled
                    col, scl, quats, eulers, opacity = parser.decode_attributes(
                        col_u,
                        scl_u,
                        rot_u,
                        extra_u=extra_u,
                        apply_exp_scale=self.use_3dgs_attributes,
                    )

                    # 3DGSモードではアルファとスケールを補正して濃さと厚みを調整
                    if self.use_3dgs_attributes:
                        if self.g3ds_scale_boost != 1.0:
                            scl = scl * self.g3ds_scale_boost

                        if opacity is not None:
                            import numpy as _np
                            opacity = _np.clip(opacity * self.g3ds_alpha_boost, 0.0, 1.0)
                            col[:, 3] = _np.clip(col[:, 3] * self.g3ds_alpha_boost, 0.0, 1.0)
                    
                    # --- 1. Binning for Render Chunks (Grid Split) ---
                    # Calculate grid indices for each point
                    grid_indices = np.floor(pos / chunk_grid_size).astype(np.int32)
                    
                    # Unique grid keys in this batch
                    unique_grids = np.unique(grid_indices, axis=0)
                    
                    for gx, gy, gz in unique_grids:
                        # Mask for points in this grid cell
                        mask = (grid_indices[:, 0] == gx) & (grid_indices[:, 1] == gy) & (grid_indices[:, 2] == gz)
                        
                        if not np.any(mask): continue
                        
                        key = (gx, gy, gz)
                        if key not in spatial_chunks:
                            spatial_chunks[key] = {'pos': [], 'col': [], 'scl': [], 'rot': [], 'lod': [], 'opa': []}
                        
                        spatial_chunks[key]['pos'].append(pos[mask])
                        spatial_chunks[key]['col'].append(col[mask])
                        spatial_chunks[key]['scl'].append(scl[mask])
                        # ビューア系は常にオイラー角を使用
                        spatial_chunks[key]['rot'].append(eulers[mask])
                        if opacity is not None:
                            spatial_chunks[key]['opa'].append(opacity[mask])
                        # Store LOD level for each point (since chunks can mix LODs now)
                        spatial_chunks[key]['lod'].append(np.full(np.sum(mask), lvl, dtype=np.int32))

                    # --- 2. Collect data for Viewport Proxy (Downsampled via Python) ---
                    # Sampling logic to avoid memory explosion for viewport object
                    sample_rate = self.viewport_density / 100.0
                    if sample_rate < 1.0:
                        # Create a boolean mask for sampling
                        count = len(pos)
                        mask = np.random.rand(count) < sample_rate
                        if np.any(mask):
                            vp_pos_list.append(pos[mask])
                            vp_col_list.append(col[mask])
                            vp_scl_list.append(scl[mask])
                            vp_rot_list.append(eulers[mask])
                            vp_rot_quat_list.append(quats[mask])
                            if opacity is not None:
                                vp_opa_list.append(opacity[mask])
                    else:
                        vp_pos_list.append(pos)
                        vp_col_list.append(col)
                        vp_scl_list.append(scl)
                        vp_rot_list.append(eulers)
                        vp_rot_quat_list.append(quats)
                        if opacity is not None:
                            vp_opa_list.append(opacity)
        
        # Apply RENDER Geometry Nodes (LOD Logic)
            # Recenter all data near origin to avoid huge world coordinates
            spatial_chunks, vp_pos_list, origin_offset = self._recenter_to_origin(
                spatial_chunks, vp_pos_list, render_coll
            )

            # Create Render Chunks from Spatial Grid
            print(f"Creating {len(spatial_chunks)} Spatial Render Chunks...")
            
            # Get Camera Location for Culling
            target_cam = bpy.data.objects.get("PanoramaCam")
            if not target_cam: target_cam = context.scene.camera
            cam_loc = target_cam.location if target_cam else Vector((0, 0, 0))

            for key, data in spatial_chunks.items():
                gx, gy, gz = key
                chunk_name = f"LCC_Render_Chunk_{gx}_{gy}_{gz}"
                
                c_pos = np.concatenate(data['pos'])
                c_col = np.concatenate(data['col'])
                c_scl = np.concatenate(data['scl'])
                c_rot = np.concatenate(data['rot'])
                c_lod = np.concatenate(data['lod'])
                c_opa = np.concatenate(data['opa']) if data['opa'] else None
                
                # Culling Logic
                should_hide = False
                if self.culling_distance > 0 and len(c_pos) > 0:
                    # Use actual point centroid (already recentered) for distance test
                    center_np = c_pos.mean(axis=0)
                    chunk_center = Vector(center_np.tolist())
                    
                    dist = (chunk_center - cam_loc).length
                    if dist > self.culling_distance:
                        should_hide = True
                
                self.create_render_chunk(render_coll, chunk_name, c_pos, c_col, c_scl, c_rot, c_lod, opacities=c_opa, hide_render_force=should_hide)

            # Create Unified Viewport Object OR GLSL Renderer
            if vp_pos_list:
                vp_pos = np.concatenate(vp_pos_list)
                vp_col = np.concatenate(vp_col_list)
                vp_scl = np.concatenate(vp_scl_list)
                vp_rot_euler = np.concatenate(vp_rot_list)
                vp_rot_quat = np.concatenate(vp_rot_quat_list)
                vp_opa = np.concatenate(vp_opa_list) if vp_opa_list else None
                
                print(f"DEBUG: use_glsl={self.use_glsl}, LCCGLSLRenderer_Class={LCCGLSLRenderer}")
                if LCCGLSLRenderer is None:
                    print(f"DEBUG: Import Error was: {import_error_msg}")
                    print(f"DEBUG: sys.path: {sys.path}")
                    # Try importing again to force error log here
                    try:
                        import LCC_GLSL_Renderer
                    except Exception as e:
                        print(f"DEBUG: Re-import attempt failed: {e}")
                        import traceback
                        traceback.print_exc()

                if self.use_glsl and LCCGLSLRenderer:
                    print(f"Initializing GLSL Renderer with {len(vp_pos)} points...")
                    global glsl_renderer, draw_handler
                    
                    # Unregister old handler if exists
                    if draw_handler:
                        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
                        draw_handler = None
                        
                    glsl_renderer = LCCGLSLRenderer()
                    glsl_renderer.load_data(vp_pos, vp_col, vp_scl, vp_rot_quat)
                    
                    # Register new handler
                    draw_handler = bpy.types.SpaceView3D.draw_handler_add(
                        self.draw_callback, (), 'WINDOW', 'POST_VIEW'
                    )
                    
                    # Force redraw
                    for area in context.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                            
                    self.report({'INFO'}, f"GLSL Renderer Initialized: {len(vp_pos)} points.")
                else:
                    print("Creating Unified Viewport Proxy Object (Mesh)...")
                    viewport_coll_name = "LCC_Viewport"
                    if viewport_coll_name not in bpy.data.collections:
                        viewport_coll = bpy.data.collections.new(viewport_coll_name)
                        context.scene.collection.children.link(viewport_coll)
                    else:
                        viewport_coll = bpy.data.collections[viewport_coll_name]
                        
                    self.create_viewport_object(viewport_coll, vp_pos, vp_col, vp_scl, vp_rot_euler, opacities=vp_opa)
                    print(f"Viewport Proxy Created: {len(vp_pos)} points.")
                
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        if total_points == 0 and not spatial_chunks and not vp_pos_list:
             # Just a warning if nothing happened
             pass
            
        return {'FINISHED'}

    def draw_callback(self):
        global glsl_renderer, last_cam_pos
        if glsl_renderer:
            try:
                # 1. Get Camera Position
                region_data = bpy.context.region_data
                if not region_data: return
                
                view_matrix = region_data.view_matrix
                # Camera position in World Space
                cam_pos = view_matrix.inverted().translation
                
                # 2. Check if sorting is needed (Distance Check)
                # Sort only if camera moved more than X units (e.g. 0.1m)
                should_sort = True
                if last_cam_pos:
                    dist = (cam_pos - last_cam_pos).length
                    if dist < 0.1: # Threshold
                        should_sort = False
                
                if should_sort:
                    glsl_renderer.sort_and_update(cam_pos)

                # Draw with the latest sorted data
                glsl_renderer.draw()
            except Exception as e:
                print(f"[LCC] Error in draw_callback: {e}")
                import traceback
                traceback.print_exc()
                
    def create_render_chunk(self, collection, chunk_name, positions, colors, scales, rots, lod_levels, opacities=None, hide_render_force=False):
        mesh = bpy.data.meshes.new(name=chunk_name)
        pos32 = np.asarray(positions, dtype=np.float32)
        num_points = len(pos32)
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", pos32.flatten())
        
        self._add_attributes(mesh, colors, scales, rots, lod_levels, opacities=opacities)
        
        obj = bpy.data.objects.new(chunk_name, mesh)
        collection.objects.link(obj)
        
        # Settings for RENDER object
        obj.hide_viewport = True  # Hidden in viewport
        obj.hide_render = hide_render_force   # Visible in render unless culled
        
        # Bounding Box Display (optional, good for debugging invisible objects)
        if self.show_bounds:
            obj.show_bounds = True
            obj.display_type = 'BOUNDS' 
        
        # Apply RENDER Geometry Nodes (LOD Logic)
        self.create_render_geonodes(obj, self.scale_density, self.min_thickness, self.lod_distance)

    def create_viewport_object(self, collection, positions, colors, scales, rots, opacities=None):
        """Creates a unified object for VIEWPORT ONLY (Hidden in Render)"""
        mesh_name = "LCC_Viewport_Proxy"
        mesh = bpy.data.meshes.new(name=mesh_name)
        
        pos32 = np.asarray(positions, dtype=np.float32)
        num_points = len(pos32)
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", pos32.flatten())
        
        # Viewport proxy doesn't need LOD level attribute essentially, but keeping structure
        self._add_attributes(mesh, colors, scales, rots, 0, opacities=opacities)
        
        obj = bpy.data.objects.new(mesh_name, mesh)
        collection.objects.link(obj)
        
        # Settings for VIEWPORT object
        obj.hide_viewport = False # Visible in viewport
        obj.hide_render = True    # Hidden in render
        
        # Apply VIEWPORT Geometry Nodes (Simple Display)
        self.create_viewport_geonodes(obj, self.scale_density, self.min_thickness)

    def _recenter_to_origin(self, spatial_chunks, vp_pos_list, render_coll):
        """Shift all positions so the scene is centered near the origin and store the offset (compute in float64, store in float32)."""
        if not spatial_chunks:
            return spatial_chunks, vp_pos_list, None

        all_pos64 = []
        for data in spatial_chunks.values():
            for arr in data['pos']:
                if arr is not None and arr.size > 0:
                    all_pos64.append(np.asarray(arr, dtype=np.float64))

        if not all_pos64:
            return spatial_chunks, vp_pos_list, None

        all_pos64 = np.concatenate(all_pos64, axis=0)
        bb_min = all_pos64.min(axis=0)
        bb_max = all_pos64.max(axis=0)
        origin_offset = (bb_min + bb_max) * 0.5

        for data in spatial_chunks.values():
            new_pos_list = []
            for arr in data['pos']:
                if arr is None or arr.size == 0:
                    new_pos_list.append(arr)
                else:
                    arr64 = np.asarray(arr, dtype=np.float64)
                    new_pos_list.append((arr64 - origin_offset).astype(np.float32))
            data['pos'] = new_pos_list

        if vp_pos_list:
            vp_pos_list = [
                (np.asarray(arr, dtype=np.float64) - origin_offset).astype(np.float32)
                for arr in vp_pos_list
            ]

        # Keep world offset so original coordinates can be recovered later
        render_coll["lcc_world_offset"] = [float(v) for v in origin_offset]

        return spatial_chunks, vp_pos_list, origin_offset

    def _add_attributes(self, mesh, colors, scales, rots, lod_data, opacities=None):
        attr_col = mesh.attributes.new(name="SplatColor", type='FLOAT_COLOR', domain='POINT')
        attr_col.data.foreach_set("color", colors.flatten())
        
        attr_scl = mesh.attributes.new(name="SplatScale", type='FLOAT_VECTOR', domain='POINT')
        attr_scl.data.foreach_set("vector", scales.flatten())
        
        attr_rot = mesh.attributes.new(name="SplatRotation", type='FLOAT_VECTOR', domain='POINT')
        attr_rot.data.foreach_set("vector", rots.flatten())

        if opacities is not None:
            attr_opa = mesh.attributes.new(name="SplatOpacity", type='FLOAT', domain='POINT')
            attr_opa.data.foreach_set("value", np.asarray(opacities, dtype=np.float32))
        
        attr_lod = mesh.attributes.new(name="SplatLOD", type='INT', domain='POINT')
        
        # Handle both single int (uniform LOD) and array (mixed LOD)
        if isinstance(lod_data, int):
             final_lod_data = np.full(len(mesh.vertices), lod_data, dtype=np.int32)
        else:
             final_lod_data = lod_data
             
        attr_lod.data.foreach_set("value", final_lod_data)

    def _add_attributes_3dgs(self, mesh, colors, scales, quats,
                             opacities=None, use_wxyz=False, add_log_opacity=True):
        """3DGS 用の Attribute を Mesh に追加する。

        作る Attribute は 3DGS PLY の必須分だけ：
          - f_dc_0, f_dc_1, f_dc_2
          - opacity
          - scale_0, scale_1, scale_2  （log-scale）
          - rot_0, rot_1, rot_2, rot_3
        """
        import numpy as _np

        Y00 = 0.28209479177387814

        # ---- 基本チェック ----
        n_verts = len(mesh.vertices)

        colors = _np.asarray(colors, dtype=_np.float32)
        scales = _np.asarray(scales, dtype=_np.float32)
        quats  = _np.asarray(quats,  dtype=_np.float32)

        if opacities is None:
            opa = _np.ones(n_verts, dtype=_np.float32)
        else:
            opa = _np.asarray(opacities, dtype=_np.float32).reshape(-1)

        # サイズ不一致なら即エラーにして場所を特定しやすくする
        if (
            colors.shape[0] != n_verts
            or scales.shape[0] != n_verts
            or quats.shape[0]  != n_verts
            or opa.shape[0]    != n_verts
        ):
            raise ValueError(
                f"3DGS attribute size mismatch: verts={n_verts}, "
                f"colors={colors.shape}, scales={scales.shape}, "
                f"quats={quats.shape}, opa={opa.shape}"
            )

        # foreach_set 用のセーフラッパ
        # attr.data の長さが 0 の場合は警告を出してスキップ
        def _safe_set(attr_name, values):
            attr = mesh.attributes.new(attr_name, 'FLOAT', 'POINT')
            data   = attr.data
            needed = len(data)
            given  = len(values)
            print(f"[LCC DEBUG] attr {attr_name}: verts={n_verts}, "
                  f"attr_len={needed}, given={given}")

            if needed == 0:
                # ここが今回のエラー原因パターン
                print(
                    f"[LCC WARNING] Attribute '{attr_name}' の長さが 0 です。"
                    "Blender が Attribute 配列の確保に失敗している可能性があります "
                    "(メモリ不足 or Blender バージョン依存)。この Attribute はスキップします。"
                )
                return

            if given != needed:
                raise ValueError(
                    f"Attribute '{attr_name}' length mismatch: verts={needed}, given={given}"
                )

            data.foreach_set("value", _np.asarray(values, dtype=_np.float32))

        # ---- DC color (f_dc_0..2) ----
        f_dc = colors[:, :3] / Y00
        _safe_set("f_dc_0", f_dc[:, 0])
        _safe_set("f_dc_1", f_dc[:, 1])
        _safe_set("f_dc_2", f_dc[:, 2])

        # ---- opacity ----
        _safe_set("opacity", opa)

        # ---- log-scale 半径 (scale_0..2) ----
        scales_log = _np.log(_np.clip(scales, 1e-8, None))
        _safe_set("scale_0", scales_log[:, 0])
        _safe_set("scale_1", scales_log[:, 1])
        _safe_set("scale_2", scales_log[:, 2])

        # ---- 回転 (quaternion) ----
        # use_wxyz=True のときは (w,x,y,z) を (x,y,z,w) に並べ替え
        if use_wxyz and quats.shape[1] >= 4:
            rot = _np.stack(
                [quats[:, 3], quats[:, 0], quats[:, 1], quats[:, 2]],
                axis=1
            )
        else:
            rot = quats

        _safe_set("rot_0", rot[:, 0])
        _safe_set("rot_1", rot[:, 1])
        _safe_set("rot_2", rot[:, 2])
        _safe_set("rot_3", rot[:, 3])


    def _create_3dgs_object(self, collection, name, positions, colors, scales, quats, opacities, origin_offset, node_group_name, use_wxyz):
        count = len(positions)
        if count == 0:
            print("[LCC] 3DGSオブジェクトに頂点がありません。スキップします。")
            return None

        mesh = bpy.data.meshes.new(name)
        mesh.vertices.add(count)
        mesh.vertices.foreach_set("co", positions.astype(np.float32).flatten())

        self._add_attributes_3dgs(mesh, colors, scales, quats, opacities=opacities, use_wxyz=use_wxyz)

        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)

        if origin_offset is not None:
            obj["lcc_offset"] = np.asarray(origin_offset, dtype=float).tolist()

        if node_group_name and node_group_name in bpy.data.node_groups:
            mod = obj.modifiers.new(name="3DGS", type='NODES')
            mod.node_group = bpy.data.node_groups[node_group_name]
        else:
            print(f"[LCC] Node group '{node_group_name}' not found. Please assign manually.")

        return obj

    def cleanup_old_objects(self):
        # Cleanup objects and collections
        collections_to_remove = ["LCC_Render", "LCC_Viewport", "LCC_Imported"]
        
        for col_name in collections_to_remove:
            if col_name in bpy.data.collections:
                coll = bpy.data.collections[col_name]
                for obj in coll.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)
                bpy.data.collections.remove(coll)
                
        # Also cleanup by name pattern just in case
        for obj in bpy.data.objects:
            if "LCC_Render_Chunk" in obj.name or "LCC_Viewport_Proxy" in obj.name:
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # Cleanup GLSL Renderer
        global draw_handler, glsl_renderer
        if draw_handler:
            bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
            draw_handler = None
        glsl_renderer = None

    # --- GEOMETRY NODES FOR RENDER (With LOD Culling) ---
    def create_render_geonodes(self, obj, density, thickness, render_dist):
        modifier = obj.modifiers.new(name="LCCSplatRenderer", type='NODES')
        group_name = "LCC_GN_Render"
        
        if group_name in bpy.data.node_groups:
            node_group = bpy.data.node_groups[group_name]
            self._build_render_geonodes_tree(node_group, density, thickness, render_dist)
        else:
            node_group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
            self._build_render_geonodes_tree(node_group, density, thickness, render_dist)
            
        modifier.node_group = node_group
        
        target_cam = bpy.data.objects.get("PanoramaCam")
        if not target_cam: target_cam = bpy.context.scene.camera
        if "Camera" in modifier.keys(): modifier["Camera"] = target_cam

    def _build_render_geonodes_tree(self, node_group, density, thickness, render_dist):
        # インターフェイス作り直し
        self._clear_interface(node_group)
        node_group.interface.new_socket(
            name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry'
        )
        node_group.interface.new_socket(
            name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry'
        )

        nodes = node_group.nodes
        links = node_group.links
        for n in list(nodes):
            nodes.remove(n)

        input_node = nodes.new('NodeGroupInput')
        input_node.location = (-1800, 0)
        output_node = nodes.new('NodeGroupOutput')
        output_node.location = (1000, 0)

        # --- 距離計算用 ---
        pos_node = nodes.new('GeometryNodeInputPosition')
        pos_node.location = (-1000, 500)

        # 1) アクティブカメラノードが使えればそれを優先
        cam_source_socket = None
        try:
            active_cam = nodes.new('GeometryNodeInputActiveCamera')
            active_cam.location = (-1200, 600)
            try:
                cam_source_socket = active_cam.outputs['Camera']
            except KeyError:
                cam_source_socket = active_cam.outputs[0]
        except Exception:
            active_cam = None
            cam_source_socket = None

        # 2) 使えない場合は従来のグループ入力にフォールバック
        if cam_source_socket is None:
            node_group.interface.new_socket(
                name="Camera", in_out='INPUT', socket_type='NodeSocketObject'
            )
            cam_source_socket = input_node.outputs['Camera']

        cam_info = nodes.new('GeometryNodeObjectInfo')
        cam_info.transform_space = 'RELATIVE'
        cam_info.location = (-1000, 600)
        links.new(cam_source_socket, cam_info.inputs[0])

        # カメラとの距離
        dist_node = nodes.new('ShaderNodeVectorMath')
        dist_node.operation = 'DISTANCE'
        dist_node.location = (-800, 500)
        links.new(pos_node.outputs['Position'], dist_node.inputs[0])
        links.new(cam_info.outputs['Location'], dist_node.inputs[1])

        # 距離を 500 で割る（Render 用スケール計算に使う）
        dist_div = nodes.new('ShaderNodeMath')
        dist_div.name = "LCC_DistanceDiv"
        dist_div.operation = 'DIVIDE'
        dist_div.inputs[1].default_value = 500.0
        dist_div.location = (-650, 350)
        links.new(dist_node.outputs['Value'], dist_div.inputs[0])

        # 描画距離判定（LOD）
        render_compare = nodes.new('FunctionNodeCompare')
        render_compare.name = "LCC_RenderDistance"
        render_compare.data_type = 'FLOAT'
        render_compare.operation = 'LESS_THAN'
        render_compare.inputs['B'].default_value = render_dist
        render_compare.location = (-600, 500)
        links.new(dist_node.outputs['Value'], render_compare.inputs['A'])

        # Instancing: 距離スケールを渡す
        self._build_instancing_part(
            nodes,
            links,
            input_node,
            output_node,
            render_compare.outputs[0],
            density,
            thickness,
            high_detail=True,
            distance_scale_socket=dist_div.outputs[0],
        )

    # --- GEOMETRY NODES FOR VIEWPORT (Simple Display) ---
    def create_viewport_geonodes(self, obj, density, thickness):
        modifier = obj.modifiers.new(name="LCCSplatViewport", type='NODES')
        group_name = "LCC_GN_Viewport"
        
        if group_name in bpy.data.node_groups:
            node_group = bpy.data.node_groups[group_name]
            self._build_viewport_geonodes_tree(node_group, density, thickness)
        else:
            node_group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
            self._build_viewport_geonodes_tree(node_group, density, thickness)
            
        modifier.node_group = node_group

    def _build_viewport_geonodes_tree(self, node_group, density, thickness):
        self._clear_interface(node_group)
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

        nodes = node_group.nodes
        links = node_group.links
        for n in nodes: nodes.remove(n)

        input_node = nodes.new('NodeGroupInput')
        input_node.location = (-1800, 0)
        output_node = nodes.new('NodeGroupOutput')
        output_node.location = (1000, 0)
        
        # No filtering selection for viewport (show all in the proxy mesh)
        self._build_instancing_part(nodes, links, input_node, output_node, None, density, thickness, high_detail=False, distance_scale_socket=None)

    # --- SHARED HELPERS ---
    def _clear_interface(self, node_group):
        if hasattr(node_group, "interface"):
             for item in list(node_group.interface.items_tree):
                 node_group.interface.remove(item)
        else:
             node_group.inputs.clear()
             node_group.outputs.clear()

    def _build_instancing_part(self, nodes, links, input_node, output_node, selection_socket, density, thickness, high_detail=False, distance_scale_socket=None):
        """Builds the core splat instancing logic shared by both Render and Viewport."""

        # Attributes
        scale_read = nodes.new('GeometryNodeInputNamedAttribute')
        scale_read.data_type = 'FLOAT_VECTOR'
        scale_read.inputs['Name'].default_value = "SplatScale"
        scale_read.location = (-1200, -100)

        rot_read = nodes.new('GeometryNodeInputNamedAttribute')
        rot_read.data_type = 'FLOAT_VECTOR'
        rot_read.inputs['Name'].default_value = "SplatRotation"
        rot_read.location = (-1200, -200)

        color_read = nodes.new('GeometryNodeInputNamedAttribute')
        color_read.data_type = 'FLOAT_COLOR'
        color_read.inputs['Name'].default_value = "SplatColor"
        color_read.location = (-1200, -300)

        # --- Scale Logic ---
        # まず SplatScale に density を掛ける（ベーススケール）
        base_scale = nodes.new('ShaderNodeVectorMath')
        base_scale.name = "LCC_ScaleBoost"
        base_scale.operation = 'SCALE'
        base_scale.inputs[3].default_value = density
        base_scale.location = (-900, 100)
        links.new(scale_read.outputs['Attribute'], base_scale.inputs[0])

        scale_for_clamp = base_scale.outputs['Vector']

        # Render 用: 距離/500 をさらにスケールとして掛ける
        if distance_scale_socket is not None:
            dist_scale = nodes.new('ShaderNodeVectorMath')
            dist_scale.name = "LCC_DistanceScale"
            dist_scale.operation = 'SCALE'
            dist_scale.location = (-700, 100)
            links.new(scale_for_clamp, dist_scale.inputs[0])
            links.new(distance_scale_socket, dist_scale.inputs[3])
            scale_for_clamp = dist_scale.outputs['Vector']

        # 最小厚みでクランプ
        scale_clamp = nodes.new('ShaderNodeVectorMath')
        scale_clamp.name = "LCC_ThicknessClamp"
        scale_clamp.operation = 'MAXIMUM'
        scale_clamp.inputs[1].default_value = (thickness, thickness, thickness)
        scale_clamp.location = (-500, 100)
        links.new(scale_for_clamp, scale_clamp.inputs[0])

        # --- SplatScale 最大値フィルタ ---
        # scale_clamp後のスケール(X/Y/Z)の最大値がしきい値以下のポイントだけ残す
        sep_scale = nodes.new('ShaderNodeSeparateXYZ')
        sep_scale.location = (-500, 300)
        links.new(scale_clamp.outputs['Vector'], sep_scale.inputs['Vector'])

        max_xy = nodes.new('ShaderNodeMath')
        max_xy.operation = 'MAXIMUM'
        max_xy.location = (-300, 320)
        links.new(sep_scale.outputs['X'], max_xy.inputs[0])
        links.new(sep_scale.outputs['Y'], max_xy.inputs[1])

        max_xyz = nodes.new('ShaderNodeMath')
        max_xyz.operation = 'MAXIMUM'
        max_xyz.location = (-120, 320)
        links.new(max_xy.outputs[0], max_xyz.inputs[0])
        links.new(sep_scale.outputs['Z'], max_xyz.inputs[1])

        scale_filter = nodes.new('FunctionNodeCompare')
        scale_filter.name = "LCC_SplatScaleFilter"
        scale_filter.data_type = 'FLOAT'
        scale_filter.operation = 'LESS_EQUAL'
        # フィルタしきい値の初期値 5
        scale_filter.inputs['B'].default_value = 5.0
        scale_filter.location = (60, 320)
        links.new(max_xyz.outputs[0], scale_filter.inputs['A'])

        # 既存の selection_socket があれば AND で合成
        final_selection = scale_filter.outputs[0]
        if selection_socket:
            bool_and = nodes.new('FunctionNodeBooleanMath')
            bool_and.operation = 'AND'
            bool_and.location = (260, 260)
            links.new(selection_socket, bool_and.inputs[0])
            links.new(scale_filter.outputs[0], bool_and.inputs[1])
            final_selection = bool_and.outputs[0]

        # Instancing
        instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
        instance_on_points.location = (0, 0)
        links.new(input_node.outputs['Geometry'], instance_on_points.inputs['Points'])

        if final_selection:
            links.new(final_selection, instance_on_points.inputs['Selection'])

        # インスタンス形状（キューブ）
        mesh_node = nodes.new('GeometryNodeMeshCube')
        mesh_node.inputs['Size'].default_value = (2.0, 2.0, 2.0)
        mesh_node.location = (-200, -200)
        links.new(mesh_node.outputs['Mesh'], instance_on_points.inputs['Instance'])

        links.new(scale_clamp.outputs['Vector'], instance_on_points.inputs['Scale'])
        links.new(rot_read.outputs['Attribute'], instance_on_points.inputs['Rotation'])

        # 色をインスタンス属性として保持
        store_color = nodes.new('GeometryNodeStoreNamedAttribute')
        store_color.data_type = 'FLOAT_COLOR'
        store_color.domain = 'INSTANCE'
        store_color.inputs['Name'].default_value = "viz_color"
        store_color.location = (200, 0)
        links.new(instance_on_points.outputs['Instances'], store_color.inputs['Geometry'])
        links.new(color_read.outputs['Attribute'], store_color.inputs['Value'])

        # マテリアル
        mat_node = nodes.new('GeometryNodeSetMaterial')
        mat_node.location = (400, 0)
        mat_name = "GaussianSplatMat"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(mat_name)
        mat_node.inputs['Material'].default_value = mat
        links.new(store_color.outputs['Geometry'], mat_node.inputs['Geometry'])

        # 出力
        links.new(mat_node.outputs['Geometry'], output_node.inputs['Geometry'])

    def _create_shader_nodes(self, mat):
        nt = mat.node_tree
        nt.nodes.clear()
        
        out_node = nt.nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (800, 0)
        
        attr_node = nt.nodes.new('ShaderNodeAttribute')
        attr_node.attribute_type = 'INSTANCER'
        attr_node.attribute_name = "viz_color"
        attr_node.location = (-800, 0)
        
        tex_coord = nt.nodes.new('ShaderNodeTexCoord')
        tex_coord.location = (-800, 300)
        
        vec_math_dot = nt.nodes.new('ShaderNodeVectorMath')
        vec_math_dot.operation = 'DOT_PRODUCT'
        vec_math_dot.location = (-600, 300)
        nt.links.new(tex_coord.outputs['Object'], vec_math_dot.inputs[0])
        nt.links.new(tex_coord.outputs['Object'], vec_math_dot.inputs[1])
        
        math_mult_factor = nt.nodes.new('ShaderNodeMath')
        math_mult_factor.name = "LCC_GaussK"
        math_mult_factor.operation = 'MULTIPLY'
        math_mult_factor.inputs[1].default_value = -1.0  # k in exp(-k * r^2), adjust if needed
        math_mult_factor.location = (-400, 300)
        nt.links.new(vec_math_dot.outputs[0], math_mult_factor.inputs[0])
        
        math_exp = nt.nodes.new('ShaderNodeMath')
        math_exp.operation = 'EXPONENT'
        math_exp.location = (-200, 300)
        nt.links.new(math_mult_factor.outputs[0], math_exp.inputs[0])

        # アルファ補正用
        alpha_boost = nt.nodes.new('ShaderNodeMath')
        alpha_boost.name = "LCC_AlphaBoost"
        alpha_boost.operation = 'MULTIPLY'
        alpha_boost.inputs[1].default_value = 1.0
        alpha_boost.location = (-200, 200)
        nt.links.new(attr_node.outputs['Alpha'], alpha_boost.inputs[0])
        
        math_mult_alpha = nt.nodes.new('ShaderNodeMath')
        math_mult_alpha.operation = 'MULTIPLY'
        math_mult_alpha.location = (0, 200)
        nt.links.new(math_exp.outputs[0], math_mult_alpha.inputs[0])
        nt.links.new(alpha_boost.outputs[0], math_mult_alpha.inputs[1])
        
        math_threshold = nt.nodes.new('ShaderNodeMath')
        math_threshold.name = "LCC_AlphaThreshold"
        math_threshold.operation = 'GREATER_THAN'
        math_threshold.inputs[1].default_value = 0.001
        math_threshold.location = (200, 300)
        nt.links.new(math_mult_alpha.outputs[0], math_threshold.inputs[0])
        
        math_masked_alpha = nt.nodes.new('ShaderNodeMath')
        math_masked_alpha.operation = 'MULTIPLY'
        math_masked_alpha.location = (400, 200)
        nt.links.new(math_mult_alpha.outputs[0], math_masked_alpha.inputs[0])
        nt.links.new(math_threshold.outputs[0], math_masked_alpha.inputs[1])

        emission = nt.nodes.new('ShaderNodeEmission')
        emission.location = (400, 0)
        nt.links.new(attr_node.outputs['Color'], emission.inputs['Color'])
        nt.links.new(math_masked_alpha.outputs[0], emission.inputs['Strength'])
        
        transparent = nt.nodes.new('ShaderNodeBsdfTransparent')
        transparent.location = (400, -100)
        
        mix_shader = nt.nodes.new('ShaderNodeMixShader')
        mix_shader.location = (600, 0)
        nt.links.new(math_masked_alpha.outputs[0], mix_shader.inputs['Fac'])
        nt.links.new(transparent.outputs['BSDF'], mix_shader.inputs[1])
        nt.links.new(emission.outputs['Emission'], mix_shader.inputs[2])
        
        nt.links.new(mix_shader.outputs['Shader'], out_node.inputs['Surface'])

    def setup_360_camera(self):
        bpy.context.scene.render.engine = 'CYCLES'
        cam_name = "PanoramaCam"
        if cam_name not in bpy.data.objects:
            cam_data = bpy.data.cameras.new(cam_name)
            cam_obj = bpy.data.objects.new(cam_name, cam_data)
            bpy.context.collection.objects.link(cam_obj)
            bpy.context.scene.camera = cam_obj
        else:
            cam_obj = bpy.data.objects[cam_name]
            cam_data = cam_obj.data
        cam_data.type = 'PANO'
        cam_data.panorama_type = 'EQUIRECTANGULAR'
        bpy.context.scene.render.resolution_x = 4096
        bpy.context.scene.render.resolution_y = 2048
        cam_obj.location = (0, 0, 0)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)

    def optimize_cycles_for_transparency(self):
        cycles = bpy.context.scene.cycles
        cycles.max_bounces = 32
        cycles.transparent_max_bounces = 64 
        cycles.transmission_bounces = 8
        preferences = bpy.context.preferences
        cycles_prefs = preferences.addons['cycles'].preferences
        if cycles_prefs.devices:
            cycles.device = 'GPU'
            try:
                cycles_prefs.compute_device_type = 'OPTIX'
            except:
                cycles_prefs.compute_device_type = 'CUDA'
            for device in cycles_prefs.devices:
                device.use = True

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class LCC_OT_render_360_preview_glsl(bpy.types.Operator):
    bl_idname = "lcc.render_360_preview_glsl"
    bl_label = "360プレビューをレンダー (GLSL)"
    bl_options = {'REGISTER'}

    width: bpy.props.IntProperty(
        name="幅",
        default=2048,
        min=256,
        max=8192,
    )
    height: bpy.props.IntProperty(
        name="高さ",
        default=1024,
        min=128,
        max=4096,
    )
    frame_start: bpy.props.IntProperty(
        name="開始フレーム",
        default=1,
        min=1,
    )
    frame_end: bpy.props.IntProperty(
        name="終了フレーム",
        default=90,
        min=1,
    )
    face_size: bpy.props.IntProperty(
        name="キューブ面サイズ",
        default=512,
        min=64,
        max=4096,
        description="キューブマップ1面の解像度。未指定時は width/4 を使用します。"
    )
    output_dir: bpy.props.StringProperty(
        name="出力ディレクトリ",
        subtype='DIR_PATH',
        default="//LCC_360_Preview/"
    )

    def execute(self, context):
        global glsl_renderer
        if glsl_renderer is None:
            self.report(
                {'ERROR'},
                "GLSL Renderer が初期化されていません。Import LCC で 'Use GLSL Viewport' を有効にしてから再実行してください。"
            )
            return {'CANCELLED'}

        scene = context.scene
        cam = scene.camera
        if cam is None:
            self.report({'ERROR'}, "シーンにアクティブなカメラがありません。")
            return {'CANCELLED'}

        width = self.width
        height = self.height
        face_size = self.face_size if self.face_size > 0 else max(64, width // 4)

        import os
        out_dir = bpy.path.abspath(self.output_dir)
        
        # 出力先ディレクトリを作成（ブレンドファイル未保存でもホームにフォールバック）
        try:
            os.makedirs(out_dir, exist_ok=True)
        except PermissionError:
            # .blend が未保存の場合などはホームディレクトリ配下に退避
            fallback_root = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.path.expanduser("~")
            out_dir = os.path.join(fallback_root, "LCC_360_Preview")
            os.makedirs(out_dir, exist_ok=True)
            print(f"[LCC] PermissionError: fallback to {out_dir}")

        offscreen = gpu.types.GPUOffScreen(face_size, face_size)

        # キューブマップの6面それぞれの向き（dir, up）
        face_defs = {
            'PX': (Vector((1, 0, 0)),  Vector((0, -1, 0))),
            'NX': (Vector((-1, 0, 0)), Vector((0, -1, 0))),
            'PY': (Vector((0, 1, 0)),  Vector((0, 0, 1))),
            'NY': (Vector((0, -1, 0)), Vector((0, 0, -1))),
            'PZ': (Vector((0, 0, 1)),  Vector((0, -1, 0))),
            'NZ': (Vector((0, 0, -1)), Vector((0, -1, 0))),
        }

        try:
            for frame in range(self.frame_start, self.frame_end + 1):
                scene.frame_set(frame)

                # カメラ行列を取得
                cam_mw = cam.matrix_world
                eye = cam_mw.translation
                basis = cam_mw.to_3x3()

                faces_np = {}

                for key, (dir_local, up_local) in face_defs.items():
                    # ビュー方向とアップベクトルをワールド座標に変換
                    dir_world = basis @ dir_local
                    up_world = basis @ up_local

                    # 向きに合わせたターゲットを算出
                    target = eye + dir_world

                    # view / projection 行列を構築
                    view_matrix = make_lookat(eye, target, up_world)
                    proj_matrix = make_perspective(
                        math.radians(90.0),  # 90度のFOV
                        aspect=1.0,
                        z_near=0.1,
                        z_far=1000.0,
                    )

                    # OffScreen に描画
                    with offscreen.bind():
                        # フレームバッファを取得
                        framebuffer = gpu.state.active_framebuffer_get()

                        # 深度書き込みやブレンドを設定
                        gpu.state.depth_mask_set(True)
                        gpu.state.depth_test_set('LESS_EQUAL')  # 'LESS' よりも少し寛容
                        gpu.state.blend_set('ALPHA_PREMULT')

                        # Blender 4.5 以降の仕様に合わせて初期化
                        framebuffer.clear(color=(0.0, 0.0, 0.0, 1.0), depth=1.0)

                        # 3DGS の描画
                        glsl_renderer.draw_offscreen(view_matrix, proj_matrix, face_size, face_size)

                        # framebuffer から RGBA を取得
                        buf = framebuffer.read_color(0, 0, face_size, face_size, 4, 0, 'UBYTE')

                    # memoryview -> bytes
                    buf_bytes = bytes(buf)

                    # numpy 配列へ変換 (H, W, 4)
                    img_np = np.frombuffer(buf_bytes, dtype=np.uint8)
                    img_np = img_np.reshape((face_size, face_size, 4))

                    # face ごとに保持
                    faces_np[key] = img_np

                # 6面を equirect へ展開
                equirect = cubemap_to_equirect(faces_np, width, height)

                # Blender Image を経由して PNG 保存
                img = bpy.data.images.new(
                    name=f"LCC_360_preview_{frame:04d}",
                    width=width,
                    height=height,
                    alpha=True,
                    float_buffer=False,
                )
                # RGBA [0,1] に正規化して pixels に書き戻す
                pixels = (equirect.astype(np.float32) / 255.0).reshape(-1)
                img.pixels = pixels.tolist()

                filepath = os.path.join(out_dir, f"frame_{frame:04d}.png")
                img.filepath_raw = filepath
                img.file_format = 'PNG'
                img.save()
                bpy.data.images.remove(img)

            self.report({'INFO'}, f"GLSL 360 Preview を {out_dir} に書き出しました。")
        finally:
            offscreen.free()

        return {'FINISHED'}


class LCC_OT_apply_display_settings(bpy.types.Operator):
    bl_idname = "lcc.apply_display_settings"
    bl_label = "LCC表示パラメータを反映"

    def execute(self, context):
        scene = context.scene
        settings = getattr(scene, "lcc_display_settings", None)
        if settings is None:
            self.report({'ERROR'}, "LCC表示設定が見つかりません。")
            return {'CANCELLED'}

        # --- Geometry Nodes の更新 ---
        for group_name in ("LCC_GN_Render", "LCC_GN_Viewport"):
            node_group = bpy.data.node_groups.get(group_name)
            if not node_group:
                continue

            nodes = node_group.nodes

            # スケール倍率（レンダー／ビューポート共通）
            scale_boost = nodes.get("LCC_ScaleBoost")
            if scale_boost and len(scale_boost.inputs) > 3:
                scale_boost.inputs[3].default_value = settings.scale_density

            # 最小厚み
            scale_clamp = nodes.get("LCC_ThicknessClamp")
            if scale_clamp and len(scale_clamp.inputs) > 1:
                t = settings.min_thickness
                scale_clamp.inputs[1].default_value = (t, t, t)

            # スケールフィルタの閾値
            scale_filter = nodes.get("LCC_SplatScaleFilter")
            if scale_filter:
                try:
                    scale_filter.inputs['B'].default_value = settings.scale_filter_max
                except KeyError:
                    if len(scale_filter.inputs) > 1:
                        scale_filter.inputs[1].default_value = settings.scale_filter_max

            # LOD距離（レンダーのみ）
            if group_name == "LCC_GN_Render":
                render_compare = nodes.get("LCC_RenderDistance")
                if render_compare:
                    try:
                        render_compare.inputs['B'].default_value = settings.lod_distance
                    except KeyError:
                        if len(render_compare.inputs) > 1:
                            render_compare.inputs[1].default_value = settings.lod_distance

        # --- マテリアルの更新 ---
        mat = bpy.data.materials.get("GaussianSplatMat")
        if mat and mat.node_tree:
            nodes = mat.node_tree.nodes

            mult_k = nodes.get("LCC_GaussK")
            if mult_k and len(mult_k.inputs) > 1:
                mult_k.inputs[1].default_value = -settings.gauss_k

            th = nodes.get("LCC_AlphaThreshold")
            if th and len(th.inputs) > 1:
                th.inputs[1].default_value = settings.alpha_threshold

            alpha_boost = nodes.get("LCC_AlphaBoost")
            if alpha_boost and len(alpha_boost.inputs) > 1:
                alpha_boost.inputs[1].default_value = settings.alpha_boost

        self.report({'INFO'}, "LCC表示パラメータを更新しました。")
        return {'FINISHED'}

class LCC_PT_Panel(bpy.types.Panel):
    bl_label = "LCCツール（ビュー/レンダー分割）"
    bl_idname = "LCC_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LCCツール'

    def draw(self, context):
        layout = self.layout
        layout.operator("import_scene.lcc", text="LCCを読み込み (.lcc)")

        col = layout.box()
        col.label(text="GLSL 360プレビュー", icon='RENDER_STILL')
        col.operator("lcc.render_360_preview_glsl", text="360プレビューをレンダー (GLSL)")

        # --- 表示パラメータ ---
        settings = getattr(context.scene, "lcc_display_settings", None)
        if settings is not None:
            col = layout.box()
            col.label(text="表示パラメータ（後から調整）", icon='GEOMETRY_NODES')
            col.prop(settings, "scale_density")
            col.prop(settings, "min_thickness")
            col.prop(settings, "lod_distance")
            col.prop(settings, "scale_filter_max")
            col.separator()
            col.prop(settings, "gauss_k")
            col.prop(settings, "alpha_threshold")
            col.prop(settings, "alpha_boost")
            col.operator("lcc.apply_display_settings", text="現在のLCCに反映")

classes = (
    LCC_DisplaySettings,
    IMPORT_OT_lcc,
    LCC_OT_render_360_preview_glsl,
    LCC_OT_apply_display_settings,
    LCC_PT_Panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # シーンに表示設定用のプロパティをぶら下げる
    bpy.types.Scene.lcc_display_settings = bpy.props.PointerProperty(
        type=LCC_DisplaySettings
    )

def unregister():
    # PointerProperty を先に削除
    if hasattr(bpy.types.Scene, "lcc_display_settings"):
        del bpy.types.Scene.lcc_display_settings

    # クラスを解除
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    # GLSL Renderer の後始末
    global draw_handler, glsl_renderer
    if draw_handler:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, 'WINDOW')
        draw_handler = None
    glsl_renderer = None

if __name__ == "__main__":
    register()
