import bpy
import os
import json
import struct
import numpy as np
import math
from mathutils import Quaternion, Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view

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

class LCCParser:
    def __init__(self, meta_path):
        self.meta_path = meta_path
        self.dir_path = os.path.dirname(meta_path)
        self.meta = {}
        self.load_meta()

    def load_meta(self):
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
            
    def get_lod_info(self, lod_level=0):
        index_path = os.path.join(self.dir_path, "Index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError("Index.bin not found")

        total_level = self.meta.get("totalLevel", 1)
        unit_entry_size = 4 + (4 + 8 + 4) * total_level
        
        units = []
        file_size = os.path.getsize(index_path)
        num_units = file_size // unit_entry_size
        
        with open(index_path, 'rb') as f:
            for i in range(num_units):
                data = f.read(unit_entry_size)
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
                
                current_pos.append(pos_data)
                current_col.append(col_data)
                current_scl.append(scl_data)
                current_rot.append(rot_data)
                current_count += count
                
                if current_count >= batch_size:
                    yield (
                        np.concatenate(current_pos),
                        np.concatenate(current_col),
                        np.concatenate(current_scl),
                        np.concatenate(current_rot)
                    )
                    current_pos = []
                    current_col = []
                    current_scl = []
                    current_rot = []
                    current_count = 0
                    
        if current_count > 0:
            yield (
                np.concatenate(current_pos),
                np.concatenate(current_col),
                np.concatenate(current_scl),
                np.concatenate(current_rot)
            )

    def decode_attributes(self, colors_u, scales_u, rots_u):
        colors = colors_u.view(dtype=np.uint8).reshape(-1, 4).astype(np.float32) / 255.0
        
        scale_meta = next((a for a in self.meta.get('attributes', []) if a['name'] == 'scale'), None)
        if scale_meta:
            s_min = np.array(scale_meta['min'], dtype=np.float32)
            s_max = np.array(scale_meta['max'], dtype=np.float32)
        else:
            s_min = np.array([0,0,0], dtype=np.float32)
            s_max = np.array([1,1,1], dtype=np.float32)
            
        scales_norm = scales_u.astype(np.float32) / 65535.0
        scales = s_min + scales_norm * (s_max - s_min)
        
        quats_np = decode_rotation(rots_u)
        eulers = quaternion_to_euler_numpy(quats_np)
            
        return colors, scales, eulers

# ------------------------------------------------------------------------
# Blender Operator
# ------------------------------------------------------------------------

class IMPORT_OT_lcc(bpy.types.Operator):
    bl_idname = "import_scene.lcc"
    bl_label = "Import LCC (Split View/Render)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    lod_min: bpy.props.IntProperty(name="Min LOD", default=0, min=0, max=5)
    lod_max: bpy.props.IntProperty(name="Max LOD", default=0, min=0, max=8)
    
    scale_density: bpy.props.FloatProperty(name="Scale Multiplier", default=1.5, min=0.1)
    min_thickness: bpy.props.FloatProperty(name="Min Thickness", default=0.05, min=0.0)
    
    chunk_size: bpy.props.FloatProperty(name="Chunk Grid Size", default=10.0, min=1.0, description="Size of the grid cell for splitting render chunks")
    
    lod_distance: bpy.props.FloatProperty(name="Render Distance", default=100.0, min=1.0, description="Points within this distance will be fully rendered")
    
    culling_distance: bpy.props.FloatProperty(name="Chunk Culling Distance", default=0.0, min=0.0, description="Chunks further than this distance will be disabled in render (0 = disabled)")
    
    # Determines how much data is sampled for the viewport proxy object
    viewport_density: bpy.props.FloatProperty(
        name="Viewport Density (%)", 
        default=1.0, 
        min=0.1, 
        max=100.0, 
        description="Percentage of points to use for the unified viewport proxy object"
    )
    
    setup_render: bpy.props.BoolProperty(name="Setup 360 Render", default=True)
    show_bounds: bpy.props.BoolProperty(name="Show Bounds", default=True)

    def execute(self, context):
        if not self.filepath.lower().endswith(".lcc"):
            self.report({'ERROR'}, "Please select an .lcc file.")
            return {'CANCELLED'}
            
        self.cleanup_old_objects()
        
        # Collections setup
        render_coll_name = "LCC_Render"
        viewport_coll_name = "LCC_Viewport"
        
        if render_coll_name not in bpy.data.collections:
            render_coll = bpy.data.collections.new(render_coll_name)
            context.scene.collection.children.link(render_coll)
        else:
            render_coll = bpy.data.collections[render_coll_name]
            
        if viewport_coll_name not in bpy.data.collections:
            viewport_coll = bpy.data.collections.new(viewport_coll_name)
            context.scene.collection.children.link(viewport_coll)
        else:
            viewport_coll = bpy.data.collections[viewport_coll_name]

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
        vp_pos_list, vp_col_list, vp_scl_list, vp_rot_list = [], [], [], []
        
        # Spatial Grid Buffer for Render Chunks
        # Key: (grid_x, grid_y, grid_z), Value: lists of (pos, col, scl, rot, lod)
        spatial_chunks = {}
        chunk_grid_size = self.chunk_size

        try:
            for lvl in range(start_lod, end_lod + 1):
                units = parser.get_lod_info(lvl)
                if not units: continue
                
                for pos, col_u, scl_u, rot_u in parser.iter_chunks(units, batch_size=1000000):
                    # Decode
                    col, scl, rot = parser.decode_attributes(col_u, scl_u, rot_u)
                    
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
                            spatial_chunks[key] = {'pos': [], 'col': [], 'scl': [], 'rot': [], 'lod': []}
                        
                        spatial_chunks[key]['pos'].append(pos[mask])
                        spatial_chunks[key]['col'].append(col[mask])
                        spatial_chunks[key]['scl'].append(scl[mask])
                        spatial_chunks[key]['rot'].append(rot[mask])
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
                            vp_rot_list.append(rot[mask])
                    else:
                        vp_pos_list.append(pos)
                        vp_col_list.append(col)
                        vp_scl_list.append(scl)
                        vp_rot_list.append(rot)
        
        # Apply RENDER Geometry Nodes (LOD Logic)
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
                
                # Culling Logic
                should_hide = False
                if self.culling_distance > 0:
                    # Calculate Chunk Center
                    center_x = (gx + 0.5) * chunk_grid_size
                    center_y = (gy + 0.5) * chunk_grid_size
                    center_z = (gz + 0.5) * chunk_grid_size
                    chunk_center = Vector((center_x, center_y, center_z))
                    
                    dist = (chunk_center - cam_loc).length
                    if dist > self.culling_distance:
                        should_hide = True
                
                self.create_render_chunk(render_coll, chunk_name, c_pos, c_col, c_scl, c_rot, c_lod, hide_render_force=should_hide)

            # Create Unified Viewport Object
            if vp_pos_list:
                print("Creating Unified Viewport Proxy Object...")
                vp_pos = np.concatenate(vp_pos_list)
                vp_col = np.concatenate(vp_col_list)
                vp_scl = np.concatenate(vp_scl_list)
                vp_rot = np.concatenate(vp_rot_list)
                
                self.create_viewport_object(viewport_coll, vp_pos, vp_col, vp_scl, vp_rot)
                print(f"Viewport Proxy Created: {len(vp_pos)} points.")
                
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        if total_points == 0:
            self.report({'WARNING'}, "No data loaded.")
            return {'CANCELLED'}
            
        self.report({'INFO'}, f"Imported {total_points} splats. {len(spatial_chunks)} Render chunks, 1 Viewport proxy.")
        return {'FINISHED'}

    def create_render_chunk(self, collection, chunk_name, positions, colors, scales, rots, lod_levels, hide_render_force=False):
        """Creates a chunk object for RENDERING ONLY (Hidden in Viewport)"""
        mesh = bpy.data.meshes.new(name=chunk_name)
        
        num_points = len(positions)
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", positions.flatten())
        
        self._add_attributes(mesh, colors, scales, rots, lod_levels)
        
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

    def create_viewport_object(self, collection, positions, colors, scales, rots):
        """Creates a unified object for VIEWPORT ONLY (Hidden in Render)"""
        mesh_name = "LCC_Viewport_Proxy"
        mesh = bpy.data.meshes.new(name=mesh_name)
        
        num_points = len(positions)
        mesh.vertices.add(num_points)
        mesh.vertices.foreach_set("co", positions.flatten())
        
        # Viewport proxy doesn't need LOD level attribute essentially, but keeping structure
        self._add_attributes(mesh, colors, scales, rots, 0)
        
        obj = bpy.data.objects.new(mesh_name, mesh)
        collection.objects.link(obj)
        
        # Settings for VIEWPORT object
        obj.hide_viewport = False # Visible in viewport
        obj.hide_render = True    # Hidden in render
        
        # Apply VIEWPORT Geometry Nodes (Simple Display)
        self.create_viewport_geonodes(obj, self.scale_density, self.min_thickness)

    def _add_attributes(self, mesh, colors, scales, rots, lod_data):
        attr_col = mesh.attributes.new(name="SplatColor", type='FLOAT_COLOR', domain='POINT')
        attr_col.data.foreach_set("color", colors.flatten())
        
        attr_scl = mesh.attributes.new(name="SplatScale", type='FLOAT_VECTOR', domain='POINT')
        attr_scl.data.foreach_set("vector", scales.flatten())
        
        attr_rot = mesh.attributes.new(name="SplatRotation", type='FLOAT_VECTOR', domain='POINT')
        attr_rot.data.foreach_set("vector", rots.flatten())
        
        attr_lod = mesh.attributes.new(name="SplatLOD", type='INT', domain='POINT')
        
        # Handle both single int (uniform LOD) and array (mixed LOD)
        if isinstance(lod_data, int):
             final_lod_data = np.full(len(mesh.vertices), lod_data, dtype=np.int32)
        else:
             final_lod_data = lod_data
             
        attr_lod.data.foreach_set("value", final_lod_data)

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
        self._clear_interface(node_group)
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Camera", in_out='INPUT', socket_type='NodeSocketObject')

        nodes = node_group.nodes
        links = node_group.links
        for n in nodes: nodes.remove(n)

        input_node = nodes.new('NodeGroupInput')
        input_node.location = (-1800, 0)
        output_node = nodes.new('NodeGroupOutput')
        output_node.location = (1000, 0)

        # --- LOD Distance Logic ---
        pos_node = nodes.new('GeometryNodeInputPosition')
        pos_node.location = (-1000, 500)
        
        cam_info = nodes.new('GeometryNodeObjectInfo')
        cam_info.transform_space = 'RELATIVE' 
        cam_info.location = (-1000, 600)
        links.new(input_node.outputs['Camera'], cam_info.inputs[0])
        
        dist_node = nodes.new('ShaderNodeVectorMath')
        dist_node.operation = 'DISTANCE'
        dist_node.location = (-800, 500)
        links.new(pos_node.outputs['Position'], dist_node.inputs[0])
        links.new(cam_info.outputs['Location'], dist_node.inputs[1])
        
        render_compare = nodes.new('FunctionNodeCompare')
        render_compare.data_type = 'FLOAT'
        render_compare.operation = 'LESS_THAN'
        render_compare.inputs['B'].default_value = render_dist
        render_compare.location = (-600, 500)
        links.new(dist_node.outputs[0], render_compare.inputs['A'])

        self._build_instancing_part(nodes, links, input_node, output_node, render_compare.outputs[0], density, thickness, high_detail=True)

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
        self._build_instancing_part(nodes, links, input_node, output_node, None, density, thickness, high_detail=False)

    # --- SHARED HELPERS ---
    def _clear_interface(self, node_group):
        if hasattr(node_group, "interface"):
             for item in list(node_group.interface.items_tree):
                 node_group.interface.remove(item)
        else:
             node_group.inputs.clear()
             node_group.outputs.clear()

    def _build_instancing_part(self, nodes, links, input_node, output_node, selection_socket, density, thickness, high_detail=False):
        """Builds the core splat instancing logic shared by both Render and Viewport"""
        
        # Attributes
        scale_read = nodes.new('GeometryNodeInputNamedAttribute')
        scale_read.data_type = 'FLOAT_VECTOR'
        scale_read.inputs['Name'].default_value = "SplatScale"
        scale_read.location = (-1000, -100)
        
        rot_read = nodes.new('GeometryNodeInputNamedAttribute')
        rot_read.data_type = 'FLOAT_VECTOR'
        rot_read.inputs['Name'].default_value = "SplatRotation"
        rot_read.location = (-1000, -200)

        color_read = nodes.new('GeometryNodeInputNamedAttribute')
        color_read.data_type = 'FLOAT_COLOR'
        color_read.inputs['Name'].default_value = "SplatColor"
        color_read.location = (-1000, -300)

        # Scale Logic
        scale_boost = nodes.new('ShaderNodeVectorMath')
        scale_boost.operation = 'SCALE'
        scale_boost.inputs[3].default_value = density
        scale_boost.location = (-600, 100)
        links.new(scale_read.outputs['Attribute'], scale_boost.inputs[0])

        scale_clamp = nodes.new('ShaderNodeVectorMath')
        scale_clamp.operation = 'MAXIMUM'
        scale_clamp.inputs[1].default_value = (thickness, thickness, thickness)
        scale_clamp.location = (-400, 100)
        links.new(scale_boost.outputs['Vector'], scale_clamp.inputs[0])

        # Instancing
        instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
        instance_on_points.location = (0, 0)
        links.new(input_node.outputs['Geometry'], instance_on_points.inputs['Points'])
        
        if selection_socket:
            links.new(selection_socket, instance_on_points.inputs['Selection'])
        
        if high_detail:
            # Use IcoSphere for better shape detail (Render)
            mesh_node = nodes.new('GeometryNodeMeshIcoSphere')
            mesh_node.inputs['Radius'].default_value = 1.0
            mesh_node.inputs['Subdivisions'].default_value = 3
        else:
            # Use Cube for performance (Viewport)
            mesh_node = nodes.new('GeometryNodeMeshCube')
            mesh_node.inputs['Size'].default_value = (2.0, 2.0, 2.0)
            
        mesh_node.location = (-200, -200)
        links.new(mesh_node.outputs['Mesh'], instance_on_points.inputs['Instance'])

        links.new(scale_clamp.outputs['Vector'], instance_on_points.inputs['Scale'])
        links.new(rot_read.outputs['Attribute'], instance_on_points.inputs['Rotation'])
        
        # Store Color
        store_color = nodes.new('GeometryNodeStoreNamedAttribute')
        store_color.data_type = 'FLOAT_COLOR'
        store_color.domain = 'INSTANCE'
        store_color.inputs['Name'].default_value = "viz_color"
        store_color.location = (200, 0)
        
        links.new(instance_on_points.outputs['Instances'], store_color.inputs['Geometry'])
        links.new(color_read.outputs['Attribute'], store_color.inputs['Value'])

        # Material
        mat_node = nodes.new('GeometryNodeSetMaterial')
        mat_node.location = (400, 0)
        
        mat_name = "GaussianSplatMat"
        if mat_name in bpy.data.materials:
            # We don't remove material here to avoid recreating it for every chunk
            mat = bpy.data.materials[mat_name]
        else:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            mat.blend_method = 'HASHED'
            if hasattr(mat, 'shadow_method'): mat.shadow_method = 'NONE'
            self._create_shader_nodes(mat)
            
        mat_node.inputs['Material'].default_value = mat
        
        links.new(store_color.outputs['Geometry'], mat_node.inputs['Geometry'])
        links.new(mat_node.outputs['Geometry'], output_node.inputs['Geometry'])

    def _create_shader_nodes(self, mat):
        nt = mat.node_tree
        for n in nt.nodes: nt.nodes.remove(n)
        
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
        math_mult_factor.operation = 'MULTIPLY'
        math_mult_factor.inputs[1].default_value = -1.5
        math_mult_factor.location = (-400, 300)
        nt.links.new(vec_math_dot.outputs[0], math_mult_factor.inputs[0])
        
        math_exp = nt.nodes.new('ShaderNodeMath')
        math_exp.operation = 'EXPONENT'
        math_exp.location = (-200, 300)
        nt.links.new(math_mult_factor.outputs[0], math_exp.inputs[0])
        
        math_mult_alpha = nt.nodes.new('ShaderNodeMath')
        math_mult_alpha.operation = 'MULTIPLY'
        math_mult_alpha.location = (0, 200)
        nt.links.new(math_exp.outputs[0], math_mult_alpha.inputs[0])
        nt.links.new(attr_node.outputs['Alpha'], math_mult_alpha.inputs[1])
        
        math_threshold = nt.nodes.new('ShaderNodeMath')
        math_threshold.operation = 'GREATER_THAN'
        math_threshold.inputs[1].default_value = 0.004
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

class LCC_PT_Panel(bpy.types.Panel):
    bl_label = "LCC Tools (Split View/Render)"
    bl_idname = "LCC_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LCC Tools'

    def draw(self, context):
        layout = self.layout
        layout.operator("import_scene.lcc", text="Import LCC (.lcc)")

classes = (IMPORT_OT_lcc, LCC_PT_Panel)

def register():
    for cls in classes:
        try: bpy.utils.unregister_class(cls)
        except: pass
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()