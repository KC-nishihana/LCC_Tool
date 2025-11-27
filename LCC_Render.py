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
                # Parse unit data for the specific LOD level
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
                
                # Copy data to ensure contiguous memory layout
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
# Frustum Culling Logic
# ------------------------------------------------------------------------

def update_frustum_culling(scene):
    cam = scene.camera
    if not cam: return
    
    # Only run if we have imported chunks
    if "LCC_Imported" not in bpy.data.collections:
        return
        
    chunks = [obj for obj in bpy.data.collections["LCC_Imported"].objects if obj.type == 'MESH']
    if not chunks: return

    for obj in chunks:
        # Simple check: project object location to camera view
        # This assumes the chunk's origin is somewhat central or indicative.
        # For better accuracy, one might check bounding box corners.
        co = obj.location
        co_ndc = world_to_camera_view(scene, cam, co)
        
        # Check if within 0..1 range (with some margin) and in front of camera (z > 0)
        margin = 0.2 
        visible = (
            -margin <= co_ndc.x <= 1.0 + margin and
            -margin <= co_ndc.y <= 1.0 + margin and
            co_ndc.z > 0
        )
        
        # Also check distance for far clipping
        if visible and co_ndc.z > cam.data.clip_end:
            visible = False
            
        obj.hide_viewport = not visible

# ------------------------------------------------------------------------
# Blender Operator
# ------------------------------------------------------------------------

class IMPORT_OT_lcc(bpy.types.Operator):
    bl_idname = "import_scene.lcc"
    bl_label = "Import LCC (Chunked + Culling)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    lod_min: bpy.props.IntProperty(name="Min LOD", default=0, min=0, max=5)
    lod_max: bpy.props.IntProperty(name="Max LOD", default=3, min=0, max=8)
    scale_density: bpy.props.FloatProperty(name="Scale Multiplier", default=1.5, min=0.1)
    min_thickness: bpy.props.FloatProperty(name="Min Thickness", default=0.05, min=0.0)
    lod_distance: bpy.props.FloatProperty(name="LOD Distance", default=20.0, min=1.0)
    setup_render: bpy.props.BoolProperty(name="Setup 360 Render", default=True)

    def create_chunk_object(self, context, chunk_id, positions, colors, scales, rots):
        """
        Creates a Mesh object for a chunk of data (vertices only).
        """
        mesh_name = f"LCC_Chunk_{chunk_id}"
        mesh = bpy.data.meshes.new(name=mesh_name)
        
        num_points = len(positions)
        mesh.vertices.add(num_points)
        
        # Set coordinates
        mesh.vertices.foreach_set("co", positions.flatten())
        
        # Set attributes
        attr_col = mesh.attributes.new(name="SplatColor", type='FLOAT_COLOR', domain='POINT')
        attr_col.data.foreach_set("color", colors.flatten())
        
        attr_scl = mesh.attributes.new(name="SplatScale", type='FLOAT_VECTOR', domain='POINT')
        attr_scl.data.foreach_set("vector", scales.flatten())
        
        attr_rot = mesh.attributes.new(name="SplatRotation", type='FLOAT_VECTOR', domain='POINT')
        attr_rot.data.foreach_set("vector", rots.flatten())
        
        # Create object
        obj = bpy.data.objects.new(mesh_name, mesh)
        
        # Link to collection
        if "LCC_Imported" not in bpy.data.collections:
            coll = bpy.data.collections.new("LCC_Imported")
            context.scene.collection.children.link(coll)
        else:
            coll = bpy.data.collections["LCC_Imported"]
            
        coll.objects.link(obj)
        
        # Apply Geometry Nodes
        self.create_geonodes(obj, self.scale_density, self.min_thickness, self.lod_distance)
        
        return obj

    def execute(self, context):
        if not self.filepath.lower().endswith(".lcc"):
            self.report({'ERROR'}, "Please select an .lcc file.")
            return {'CANCELLED'}
            
        self.cleanup_old_objects()

        parser = LCCParser(self.filepath)
        
        target_lod_start = self.lod_min
        target_lod_end = self.lod_max
        
        start_lod = min(target_lod_start, target_lod_end)
        end_lod = max(target_lod_start, target_lod_end)
        
        total_points = 0
        chunk_id = 0
        
        try:
            # Collect all units first to pass to iter_chunks
            all_units = []
            for lvl in range(start_lod, end_lod + 1):
                units = parser.get_lod_info(lvl)
                if units:
                    all_units.extend(units)
            
            if not all_units:
                self.report({'WARNING'}, "No data found for selected LODs.")
                return {'CANCELLED'}

            # Iterate chunks
            for pos, col_u, scl_u, rot_u in parser.iter_chunks(all_units, batch_size=1000000):
                # Decode attributes for this chunk
                col, scl, rot = parser.decode_attributes(col_u, scl_u, rot_u)
                
                # Create object
                self.create_chunk_object(context, chunk_id, pos, col, scl, rot)
                
                count = len(pos)
                total_points += count
                print(f"Created Chunk {chunk_id}: {count} points")
                chunk_id += 1
                
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}

        if total_points == 0:
            self.report({'WARNING'}, "No data loaded.")
            return {'CANCELLED'}

        if self.setup_render:
            self.setup_360_camera()
            self.optimize_cycles_for_transparency()
            
        self.report({'INFO'}, f"Imported {total_points} splats in {chunk_id} chunks.")
        return {'FINISHED'}

    def cleanup_old_objects(self):
        for obj in bpy.data.objects:
            if "LCC_Splats" in obj.name or "LCC_Chunk_" in obj.name:
                bpy.data.objects.remove(obj, do_unlink=True)
        
        # Clean up collection if empty
        if "LCC_Imported" in bpy.data.collections:
            coll = bpy.data.collections["LCC_Imported"]
            if not coll.objects:
                bpy.data.collections.remove(coll)

    def create_geonodes(self, obj, density, thickness, lod_dist_factor):
        modifier = obj.modifiers.new(name="LCCSplatRenderer", type='NODES')
        
        # Check if node group exists to reuse it (optimization)
        group_name = "LCC_GN"
        if group_name in bpy.data.node_groups:
            node_group = bpy.data.node_groups[group_name]
        else:
            node_group = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
            self._build_geonodes_tree(node_group, density, thickness, lod_dist_factor)
            
        modifier.node_group = node_group

    def _build_geonodes_tree(self, node_group, density, thickness, lod_dist_factor):
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

        nodes = node_group.nodes
        links = node_group.links
        for n in nodes: nodes.remove(n)

        input_node = nodes.new('NodeGroupInput')
        input_node.location = (-1400, 0)
        output_node = nodes.new('NodeGroupOutput')
        output_node.location = (1400, 0)

        # --- VIEWPORT OPTIMIZATION ---
        is_viewport = nodes.new('GeometryNodeIsViewport')
        is_viewport.location = (-1200, 200)
        
        viewport_ratio = nodes.new('FunctionNodeRandomValue')
        viewport_ratio.data_type = 'BOOLEAN'
        viewport_ratio.inputs['Probability'].default_value = 0.1
        viewport_ratio.location = (-1200, 100)
        
        logic_not_viewport = nodes.new('FunctionNodeBooleanMath')
        logic_not_viewport.operation = 'NOT'
        logic_not_viewport.location = (-1000, 200)
        links.new(is_viewport.outputs[0], logic_not_viewport.inputs[0])
        
        logic_or = nodes.new('FunctionNodeBooleanMath')
        logic_or.operation = 'OR'
        logic_or.location = (-800, 150)
        links.new(logic_not_viewport.outputs[0], logic_or.inputs[0])
        links.new(viewport_ratio.outputs[3], logic_or.inputs[1])
        
        separate_geo = nodes.new('GeometryNodeSeparateGeometry')
        separate_geo.location = (-600, 0)
        links.new(input_node.outputs['Geometry'], separate_geo.inputs['Geometry'])
        links.new(logic_or.outputs[0], separate_geo.inputs['Selection'])
        
        current_geo = separate_geo.outputs['Selection']

        # --- Attributes ---
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
        
        lod_read = nodes.new('GeometryNodeInputNamedAttribute')
        lod_read.data_type = 'INT'
        lod_read.inputs['Name'].default_value = "SplatLOD"
        lod_read.location = (-1000, -400)

        # --- LOD Logic ---
        pos_node = nodes.new('GeometryNodeInputPosition')
        pos_node.location = (-1000, 500)
        
        cam_info = nodes.new('GeometryNodeObjectInfo')
        cam_info.inputs[0].default_value = bpy.data.objects.get("PanoramaCam") 
        cam_info.transform_space = 'RELATIVE' 
        cam_info.location = (-1000, 600)
        
        dist_node = nodes.new('ShaderNodeVectorMath')
        dist_node.operation = 'DISTANCE'
        dist_node.location = (-800, 500)
        links.new(pos_node.outputs['Position'], dist_node.inputs[0])
        links.new(cam_info.outputs['Location'], dist_node.inputs[1])
        
        math_pow = nodes.new('ShaderNodeMath')
        math_pow.operation = 'POWER'
        math_pow.inputs[0].default_value = 2.0
        math_pow.location = (-800, 300)
        links.new(lod_read.outputs['Attribute'], math_pow.inputs[1])
        
        math_mult_dist = nodes.new('ShaderNodeMath')
        math_mult_dist.operation = 'MULTIPLY'
        math_mult_dist.inputs[0].default_value = lod_dist_factor * 2.0
        math_mult_dist.location = (-600, 300)
        links.new(math_pow.outputs[0], math_mult_dist.inputs[1])
        
        rand_val = nodes.new('FunctionNodeRandomValue')
        rand_val.data_type = 'FLOAT'
        rand_val.location = (-600, 400)
        
        math_div = nodes.new('ShaderNodeMath')
        math_div.operation = 'DIVIDE'
        math_div.location = (-600, 500)
        links.new(math_mult_dist.outputs[0], math_div.inputs[0]) 
        links.new(dist_node.outputs[0], math_div.inputs[1]) 
        
        math_compare = nodes.new('FunctionNodeCompare')
        math_compare.data_type = 'FLOAT'
        math_compare.operation = 'LESS_THAN'
        math_compare.location = (-400, 400)
        links.new(rand_val.outputs['Value'], math_compare.inputs['A'])
        links.new(math_div.outputs[0], math_compare.inputs['B'])

        # --- Scale Logic ---
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

        # --- Instancing ---
        instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
        instance_on_points.location = (-200, 0)
        links.new(current_geo, instance_on_points.inputs['Points'])
        links.new(math_compare.outputs['Result'], instance_on_points.inputs['Selection'])
        
        cube_node = nodes.new('GeometryNodeMeshCube')
        cube_node.inputs['Size'].default_value = (2.0, 2.0, 2.0) 
        cube_node.location = (-400, -200)
        links.new(cube_node.outputs['Mesh'], instance_on_points.inputs['Instance'])

        links.new(scale_clamp.outputs['Vector'], instance_on_points.inputs['Scale'])
        links.new(rot_read.outputs['Attribute'], instance_on_points.inputs['Rotation'])
        
        store_color = nodes.new('GeometryNodeStoreNamedAttribute')
        store_color.data_type = 'FLOAT_COLOR'
        store_color.domain = 'INSTANCE'
        store_color.inputs['Name'].default_value = "viz_color"
        store_color.location = (0, 0)
        
        links.new(instance_on_points.outputs['Instances'], store_color.inputs['Geometry'])
        links.new(color_read.outputs['Attribute'], store_color.inputs['Value'])

        # Material
        mat_node = nodes.new('GeometryNodeSetMaterial')
        mat_node.location = (200, 0)
        
        mat_name = "GaussianSplatMat"
        if mat_name in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[mat_name])
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat.blend_method = 'HASHED'
        if hasattr(mat, 'shadow_method'): mat.shadow_method = 'NONE'
        
        nt = mat.node_tree
        for n in nt.nodes: nt.nodes.remove(n)
        
        out_node = nt.nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (800, 0)
        
        attr_node = nt.nodes.new('ShaderNodeAttribute')
        attr_node.attribute_type = 'INSTANCER' 
        attr_node.attribute_name = "viz_color" 
        attr_node.location = (-800, 0)
        
        # Shader Graph
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
        mat_node.inputs['Material'].default_value = mat
        
        links.new(store_color.outputs['Geometry'], mat_node.inputs['Geometry'])
        links.new(mat_node.outputs['Geometry'], output_node.inputs['Geometry'])

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
    bl_label = "LCC Tools (Chunked + Culling)"
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
    
    if update_frustum_culling not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(update_frustum_culling)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    
    if update_frustum_culling in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(update_frustum_culling)

if __name__ == "__main__":
    register()