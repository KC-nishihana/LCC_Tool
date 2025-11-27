import bpy
import os
import json
import struct
import numpy as np
import math
from mathutils import Quaternion, Vector, Matrix

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
        # Unit Entry Size calculation based on white paper structure
        unit_entry_size = 4 + (4 + 8 + 4) * total_level
        
        units = []
        file_size = os.path.getsize(index_path)
        num_units = file_size // unit_entry_size
        
        print(f"LCC Info: Found {num_units} units, {total_level} levels.")

        with open(index_path, 'rb') as f:
            for i in range(num_units):
                data = f.read(unit_entry_size)
                unit_idx = struct.unpack('<I', data[0:4])[0]
                
                offset_in_entry = 4 + lod_level * 16
                pc, lo, ls = struct.unpack('<IQI', data[offset_in_entry : offset_in_entry+16])
                
                if pc > 0:
                    units.append({
                        'unit_index': unit_idx,
                        'count': pc,
                        'offset': lo,
                        'size': ls
                    })
        return units

    def read_splat_data(self, units):
        data_path = os.path.join(self.dir_path, "Data.bin")
        if not os.path.exists(data_path):
            raise FileNotFoundError("Data.bin not found")

        total_splats = sum(u['count'] for u in units)
        print(f"Reading {total_splats} splats from Data.bin...")

        positions = np.zeros((total_splats, 3), dtype=np.float32)
        colors_uint = np.zeros((total_splats,), dtype=np.uint32)
        scales_uint = np.zeros((total_splats, 3), dtype=np.uint16)
        rots_uint = np.zeros((total_splats,), dtype=np.uint32)
        
        current_idx = 0
        stride = 32
        
        with open(data_path, 'rb') as f:
            for u in units:
                f.seek(u['offset'])
                raw_bytes = f.read(u['size'])
                
                count = u['count']
                expected_size = count * stride
                if len(raw_bytes) < expected_size:
                    print(f"Warning: Unit data smaller than expected. Skipping.")
                    continue
                
                chunk_data = np.frombuffer(raw_bytes[:expected_size], dtype=np.uint8)
                chunk_matrix = chunk_data.reshape((count, 32))
                
                # Position (0-12 bytes)
                pos_bytes = chunk_matrix[:, 0:12].tobytes()
                positions[current_idx : current_idx+count] = np.frombuffer(pos_bytes, dtype=np.float32).reshape((count, 3))
                
                # Color (12-16 bytes)
                col_bytes = chunk_matrix[:, 12:16].tobytes()
                colors_uint[current_idx : current_idx+count] = np.frombuffer(col_bytes, dtype=np.uint32)
                
                # Scale (16-22 bytes)
                scl_bytes = chunk_matrix[:, 16:22].tobytes()
                scales_uint[current_idx : current_idx+count] = np.frombuffer(scl_bytes, dtype=np.uint16).reshape((count, 3))
                
                # Rotation (22-26 bytes)
                rot_bytes = chunk_matrix[:, 22:26].tobytes()
                rots_uint[current_idx : current_idx+count] = np.frombuffer(rot_bytes, dtype=np.uint32)
                
                current_idx += count

        return positions, colors_uint, scales_uint, rots_uint

    def decode_attributes(self, colors_u, scales_u, rots_u):
        count = len(colors_u)
        
        # Color
        colors_bytes = colors_u.view(dtype=np.uint8).reshape((count, 4))
        colors = colors_bytes.astype(np.float32) / 255.0
        
        if count > 0:
            avg_color = np.mean(colors, axis=0)
            print(f"Debug: Loaded {count} colors. Average RGBA: {avg_color}")
        
        # Scale
        scale_meta = next((a for a in self.meta.get('attributes', []) if a['name'] == 'scale'), None)
        if scale_meta:
            s_min = np.array(scale_meta['min'], dtype=np.float32)
            s_max = np.array(scale_meta['max'], dtype=np.float32)
        else:
            s_min = np.array([0,0,0], dtype=np.float32)
            s_max = np.array([1,1,1], dtype=np.float32)
            
        scales_norm = scales_u.astype(np.float32) / 65535.0
        scales = s_min + scales_norm * (s_max - s_min)
        
        # Rotation
        quats_np = decode_rotation(rots_u)
        
        print("Converting rotations to Euler...")
        eulers = np.zeros((count, 3), dtype=np.float32)
        
        for i in range(count):
            x, y, z, w = quats_np[i]
            q = Quaternion((w, x, y, z))
            e = q.to_euler()
            eulers[i, 0] = e.x
            eulers[i, 1] = e.y
            eulers[i, 2] = e.z
            
        return colors, scales, eulers

# ------------------------------------------------------------------------
# Blender Operator
# ------------------------------------------------------------------------

class IMPORT_OT_lcc(bpy.types.Operator):
    bl_idname = "import_scene.lcc"
    bl_label = "Import LCC"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    # Import Options
    lod_level: bpy.props.IntProperty(
        name="LOD Level", 
        default=2, 
        min=0, 
        max=5, 
        description="0 is highest detail. Warning: LOD0 can be very heavy."
    )
    
    scale_density: bpy.props.FloatProperty(
        name="Splat Scale Multiplier", 
        default=1.5, 
        min=0.1, 
        max=10.0, 
        description="Multiplies the size of all splats. Higher values fill gaps but may look blobbier."
    )
    
    min_thickness: bpy.props.FloatProperty(
        name="Min Thickness", 
        default=0.05, 
        min=0.0, 
        max=1.0, 
        description="Minimum size for any axis. Prevents walls from disappearing when viewed from the side."
    )

    setup_render: bpy.props.BoolProperty(name="Setup 360 Render", default=True)

    def execute(self, context):
        if not self.filepath.lower().endswith(".lcc"):
            self.report({'ERROR'}, "Please select an .lcc file.")
            return {'CANCELLED'}

        parser = LCCParser(self.filepath)
        
        try:
            units = parser.get_lod_info(self.lod_level)
            if not units:
                self.report({'WARNING'}, f"No data found for LOD {self.lod_level}.")
                return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error parsing Index: {str(e)}")
            return {'CANCELLED'}

        try:
            pos, col_u, scl_u, rot_u = parser.read_splat_data(units)
        except Exception as e:
            self.report({'ERROR'}, f"Error reading Data: {str(e)}")
            return {'CANCELLED'}

        col, scl, eul = parser.decode_attributes(col_u, scl_u, rot_u)

        mesh_name = "LCC_Splats"
        mesh = bpy.data.meshes.new(mesh_name)
        obj = bpy.data.objects.new(mesh_name, mesh)
        context.collection.objects.link(obj)

        mesh.vertices.add(len(pos))
        mesh.vertices.foreach_set("co", pos.flatten())
        
        # Store attributes on Point domain
        color_attr = mesh.attributes.new(name="SplatColor", type='FLOAT_COLOR', domain='POINT')
        color_attr.data.foreach_set("color", col.flatten())

        scale_attr = mesh.attributes.new(name="SplatScale", type='FLOAT_VECTOR', domain='POINT')
        scale_attr.data.foreach_set("vector", scl.flatten())
        
        rot_attr = mesh.attributes.new(name="SplatRotation", type='FLOAT_VECTOR', domain='POINT')
        rot_attr.data.foreach_set("vector", eul.flatten())

        mesh.update()

        # Pass user settings to GeoNodes creator
        self.create_geonodes(obj, self.scale_density, self.min_thickness)

        if self.setup_render:
            self.setup_360_camera()
            self.optimize_cycles_for_transparency()
            
        self.report({'INFO'}, f"Imported {len(pos)} splats.")
        return {'FINISHED'}

    def create_geonodes(self, obj, density, thickness):
        modifier = obj.modifiers.new(name="LCCSplatRenderer", type='NODES')
        node_group = bpy.data.node_groups.new("LCC_GN", "GeometryNodeTree")
        modifier.node_group = node_group
        
        node_group.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_group.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')

        nodes = node_group.nodes
        links = node_group.links
        for n in nodes: nodes.remove(n)

        # 1. Inputs
        input_node = nodes.new('NodeGroupInput')
        input_node.location = (-1000, 0)
        output_node = nodes.new('NodeGroupOutput')
        output_node.location = (1000, 0)

        # 2. Read Attributes from Points
        scale_read = nodes.new('GeometryNodeInputNamedAttribute')
        scale_read.data_type = 'FLOAT_VECTOR'
        scale_read.inputs['Name'].default_value = "SplatScale"
        scale_read.location = (-800, 100)
        
        rot_read = nodes.new('GeometryNodeInputNamedAttribute')
        rot_read.data_type = 'FLOAT_VECTOR'
        rot_read.inputs['Name'].default_value = "SplatRotation"
        rot_read.location = (-800, 0)

        color_read = nodes.new('GeometryNodeInputNamedAttribute')
        color_read.data_type = 'FLOAT_COLOR'
        color_read.inputs['Name'].default_value = "SplatColor"
        color_read.location = (-800, -100)

        # --- SCALE ADJUSTMENTS (User Controlled) ---
        # 1. Boost overall size (Density)
        scale_boost = nodes.new('ShaderNodeVectorMath')
        scale_boost.operation = 'SCALE'
        scale_boost.inputs[3].default_value = density # Applied from user input
        scale_boost.location = (-600, 100)
        links.new(scale_read.outputs['Attribute'], scale_boost.inputs[0])

        # 2. Ensure Minimum Thickness (Fix for invisible walls)
        scale_clamp = nodes.new('ShaderNodeVectorMath')
        scale_clamp.operation = 'MAXIMUM'
        scale_clamp.inputs[1].default_value = (thickness, thickness, thickness) # Applied from user input
        scale_clamp.location = (-400, 100)
        links.new(scale_boost.outputs['Vector'], scale_clamp.inputs[0])

        # 3. Instance on Points
        instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
        instance_on_points.location = (-200, 0)
        
        # Use Ico Sphere instead of Grid to give volume (ellipsoids)
        sphere_node = nodes.new('GeometryNodeMeshIcoSphere')
        sphere_node.inputs['Radius'].default_value = 1.0 
        sphere_node.inputs['Subdivisions'].default_value = 1
        sphere_node.location = (-400, -200)

        # 4. Store Color on Instances
        store_color = nodes.new('GeometryNodeStoreNamedAttribute')
        store_color.data_type = 'FLOAT_COLOR'
        store_color.domain = 'INSTANCE'
        store_color.inputs['Name'].default_value = "viz_color"
        store_color.location = (0, 0)

        # 5. Realize Instances
        realize_instances = nodes.new('GeometryNodeRealizeInstances')
        realize_instances.location = (200, 0)

        # 6. Set Material
        mat_node = nodes.new('GeometryNodeSetMaterial')
        mat_node.location = (400, 0)
        
        # --- Create/Reset Material ---
        mat_name = "GaussianSplatMat"
        if mat_name in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[mat_name])
        
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        mat.blend_method = 'HASHED'
        if hasattr(mat, 'shadow_method'):
            mat.shadow_method = 'NONE'
        
        nt = mat.node_tree
        for n in nt.nodes: nt.nodes.remove(n)
        
        # Shader Nodes
        out_node = nt.nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (800, 0)
        
        attr_node = nt.nodes.new('ShaderNodeAttribute')
        attr_node.attribute_name = "viz_color" 
        attr_node.location = (-800, 0)
        
        # --- Gaussian Falloff ---
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
        
        # --- GeoNode Links ---
        links.new(input_node.outputs['Geometry'], instance_on_points.inputs['Points'])
        links.new(sphere_node.outputs['Mesh'], instance_on_points.inputs['Instance'])
        
        # Apply Clamped Scale
        links.new(scale_clamp.outputs['Vector'], instance_on_points.inputs['Scale'])
        
        links.new(rot_read.outputs['Attribute'], instance_on_points.inputs['Rotation'])
        
        links.new(instance_on_points.outputs['Instances'], store_color.inputs['Geometry'])
        links.new(color_read.outputs['Attribute'], store_color.inputs['Value'])
        
        links.new(store_color.outputs['Geometry'], realize_instances.inputs['Geometry'])
        links.new(realize_instances.outputs['Geometry'], mat_node.inputs['Geometry'])
        links.new(mat_node.outputs['Geometry'], output_node.inputs['Geometry'])

    def setup_360_camera(self):
        bpy.context.scene.render.engine = 'CYCLES'
        
        cam_data = bpy.data.cameras.new("PanoramaCam")
        cam_obj = bpy.data.objects.new("PanoramaCam", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj
        
        cam_data.type = 'PANO'
        cam_data.panorama_type = 'EQUIRECTANGULAR'
        
        bpy.context.scene.render.resolution_x = 4096
        bpy.context.scene.render.resolution_y = 2048
        
        cam_obj.location = (0, 0, 0)
        cam_obj.rotation_euler = (math.radians(90), 0, 0)

    def optimize_cycles_for_transparency(self):
        cycles = bpy.context.scene.cycles
        cycles.max_bounces = 24
        cycles.transparent_max_bounces = 32
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

# ------------------------------------------------------------------------
# UI Panel
# ------------------------------------------------------------------------

class LCC_PT_Panel(bpy.types.Panel):
    bl_label = "LCC Tools"
    bl_idname = "LCC_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'LCC Tools'

    def draw(self, context):
        layout = self.layout
        layout.operator("import_scene.lcc", text="Import LCC (.lcc)")

# ------------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------------

classes = (IMPORT_OT_lcc, LCC_PT_Panel)

def register():
    # Unregister existing classes first to avoid errors
    for cls in classes:
        try:
            bpy.utils.unregister_class(cls)
        except:
            pass
    
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()