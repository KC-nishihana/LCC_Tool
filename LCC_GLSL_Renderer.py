import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from mathutils import Matrix, Vector

# --------------------------------------------------------------------
# GLSL Shaders
# --------------------------------------------------------------------

VERT_SHADER = """
uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec3 camPos;
uniform vec2 viewportSize;

in vec3 pos;
in vec4 color;
in vec3 scale;
in vec4 rot; // Quaternion (x, y, z, w)

out vec4 vColor;
out vec2 vCenter;
out vec2 vPos;

// Helper to construct rotation matrix from quaternion
mat3 quatToMat3(vec4 q) {
    float qx = q.x; float qy = q.y; float qz = q.z; float qw = q.w;
    float x2 = qx + qx; float y2 = qy + qy; float z2 = qz + qz;
    float xx = qx * x2; float xy = qx * y2; float xz = qx * z2;
    float yy = qy * y2; float yz = qy * z2; float zz = qz * z2;
    float wx = qw * x2; float wy = qw * y2; float wz = qw * z2;

    return mat3(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy)
    );
}

void main() {
    // 1. Compute 3D Covariance Matrix
    mat3 R = quatToMat3(rot);
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    mat3 M = R * S;
    mat3 V_cov = M * transpose(M); // 3D Covariance

    // 2. Project to 2D (EWA Splatting approximation)
    // We need the view matrix (World to Camera) to transform the covariance
    // Blender's viewMatrix is WorldToCamera.
    
    // Transform center to camera space
    vec4 pos_cam = viewMatrix * vec4(pos, 1.0);
    
    // Jacobian of the perspective projection
    // J = [ 1/z  0   -x/z^2 ]
    //     [ 0   1/z  -y/z^2 ]
    // (ignoring the third row for 2D projection)
    
    // However, a simpler approximation often used in 3DGS implementations:
    // T = W * J * V * W^T * J^T
    // Here we will use a simplified billboard approach for the first step to ensure it works,
    // then upgrade to full covariance projection if needed.
    
    // --- Simplified Billboard Approach for Initial Implementation ---
    // This is not "true" 3DGS EWA splatting but gives a visual representation.
    // To do true EWA, we need more complex math in GLSL or pre-computed in Python.
    // Given the constraints, let's try to approximate the screen space size.
    
    // Let's use the standard billboard quad vertices passed as 'pos' in the batch?
    // No, we are using instancing. 'pos' is the center of the splat.
    // We need the quad vertex relative to the center.
    // But wait, we don't have the quad vertex in 'in' variables yet.
    // We usually pass the quad vertex as another attribute or use gl_VertexID.
    
    // Let's assume we pass 'vPos' (quad vertex) as a separate attribute or uniform?
    // In batch_for_shader with 'TRI_FAN', we define the geometry.
    // If we use instancing, we need to access the instance data.
    
    // Actually, for instancing in Blender Python GPU module, it's often easier to 
    // just pass the quad vertices and use 'glVertexID' to determine which corner it is,
    // OR use the 'gpu.types.GPUVertBuf' directly.
    
    // Let's stick to the user's plan: "Instancing drawing (draw 1 Quad for number of points)".
    // We will use 'gl_InstanceID' to fetch data if we use TBOs, but here we are using attributes.
    // When using 'batch_for_shader' with instancing, the attributes are per-instance.
    // The per-vertex data (the quad) needs to be separate.
    
    // But 'batch_for_shader' in Blender doesn't easily support mixing per-vertex and per-instance attributes 
    // in a single simple call without some setup.
    // A common workaround in Blender Python is to repeat the data (expensive) OR use a geometry shader (not always available/easy)
    // OR use the 'instanced' parameter in 'batch.draw(shader)'.
    
    // Let's assume we are sending the quad vertices as a base, and the rest are instance attributes.
    // In the vertex shader:
    // 'pos', 'color', 'scale', 'rot' are INSTANCE attributes.
    // We need a way to know the vertex position of the quad.
    // Let's add 'quad_pos' as an input.
}
"""

# Re-writing Vertex Shader for a working implementation
VERT_SHADER_FULL = """
uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform vec3 camPos;
uniform vec2 viewportSize;

in vec3 pos;          // Instance Attribute: Center position
in vec4 color;        // Instance Attribute: Color
in vec3 scale;        // Instance Attribute: Scale
in vec4 rot;          // Instance Attribute: Rotation (Quaternion)
in vec2 quad_pos;     // Vertex Attribute: Quad vertex (-1 to 1)

out vec4 vColor;
out vec2 vUv;

mat3 quatToMat3(vec4 q) {
    float qx = q.x; float qy = q.y; float qz = q.z; float qw = q.w;
    float x2 = qx + qx; float y2 = qy + qy; float z2 = qz + qz;
    float xx = qx * x2; float xy = qx * y2; float xz = qx * z2;
    float yy = qy * y2; float yz = qy * z2; float zz = qz * z2;
    float wx = qw * x2; float wy = qw * y2; float wz = qw * z2;
    return mat3(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy)
    );
}

void main() {
    vColor = color;
    vUv = quad_pos;

    // Construct local transformation
    mat3 R = quatToMat3(rot);
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    
    // Billboard logic:
    // We want the quad to face the camera. 
    // But 3DGS splats are 3D ellipsoids projected.
    // For a basic implementation, let's just project the ellipsoid basis vectors to screen.
    
    // Simplified approach:
    // 1. Calculate world position of the quad vertex based on local basis
    // This draws the actual 3D ellipsoid (as a flat quad in 3D space? No, that's not right).
    
    // Correct approach for 3DGS is usually:
    // Compute 2D covariance matrix in screen space, then determine the axes and size of the 2D ellipse.
    // Then vertex position = center_screen + quad_pos * axis_vectors.
    
    // Let's try a simpler billboard approach first to get pixels on screen.
    // World Space Quad oriented to camera?
    vec3 forward = normalize(camPos - pos);
    vec3 right = normalize(cross(vec3(0.0, 0.0, 1.0), forward));
    vec3 up = cross(forward, right);
    
    // Use the maximum scale component for size estimation
    float max_scale = max(scale.x, max(scale.y, scale.z));
    
    vec3 world_vert = pos + (right * quad_pos.x * max_scale) + (up * quad_pos.y * max_scale);
    
    gl_Position = viewProjectionMatrix * vec4(world_vert, 1.0);
}
"""

# Better Vertex Shader attempting 2D projection approximation
VERT_SHADER_COV = """
uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 viewportSize;
uniform float focal_x;
uniform float focal_y;

in vec3 pos;          // Instance Attribute
in vec4 color;        // Instance Attribute
in vec3 scale;        // Instance Attribute
in vec4 rot;          // Instance Attribute
in vec2 quad_pos;     // Vertex Attribute

out vec4 vColor;
out vec2 vUv;

mat3 quatToMat3(vec4 q) {
    float qx = q.x; float qy = q.y; float qz = q.z; float qw = q.w;
    float x2 = qx + qx; float y2 = qy + qy; float z2 = qz + qz;
    float xx = qx * x2; float xy = qx * y2; float xz = qx * z2;
    float yy = qy * y2; float yz = qy * z2; float zz = qz * z2;
    float wx = qw * x2; float wy = qw * y2; float wz = qw * z2;
    return mat3(
        1.0 - (yy + zz), xy - wz, xz + wy,
        xy + wz, 1.0 - (xx + zz), yz - wx,
        xz - wy, yz + wx, 1.0 - (xx + yy)
    );
}

void main() {
    vColor = color;
    vUv = quad_pos; // -1 to 1

    vec4 pos_cam = viewMatrix * vec4(pos, 1.0);
    
    // Culling: if behind camera
    if (pos_cam.z > 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // Clip
        return;
    }

    // 3D Covariance
    mat3 R = quatToMat3(rot);
    mat3 S = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    mat3 M = R * S;
    mat3 V = M * transpose(M);
    
    // Project to 2D
    // J matrix approximation
    float z = pos_cam.z;
    float x = pos_cam.x;
    float y = pos_cam.y;
    
    // Focal lengths (approximate from projection matrix)
    // proj[0][0] = 2n / (r-l) = 2 * fx / w
    // We can just use the uniform values passed in
    
    mat3 J = mat3(
        focal_x / z, 0.0, -(focal_x * x) / (z * z),
        0.0, focal_y / z, -(focal_y * y) / (z * z),
        0.0, 0.0, 0.0
    );
    
    // Transform Covariance to Ray Space (approx)
    // W is the view matrix rotation part
    mat3 W = mat3(viewMatrix);
    mat3 T = W * V * transpose(W);
    
    // 2D Covariance
    mat3 V2d = J * T * transpose(J);
    
    // Covariance 2x2
    float cov00 = V2d[0][0];
    float cov01 = V2d[0][1];
    float cov11 = V2d[1][1];
    
    // Eigendecomposition to find axes
    float lambda1 = 0.5 * (cov00 + cov11 + sqrt((cov00 - cov11) * (cov00 - cov11) + 4.0 * cov01 * cov01));
    float lambda2 = 0.5 * (cov00 + cov11 - sqrt((cov00 - cov11) * (cov00 - cov11) + 4.0 * cov01 * cov01));
    
    // Radius (3 sigma)
    float r1 = 3.0 * sqrt(max(lambda1, 0.0));
    float r2 = 3.0 * sqrt(max(lambda2, 0.0));
    
    // Angle
    float theta = 0.0;
    if (abs(cov01) > 0.0001) {
        theta = 0.5 * atan(2.0 * cov01, cov00 - cov11);
    } else if (cov00 < cov11) {
        theta = 1.570796; // 90 deg
    }
    
    float cos_t = cos(theta);
    float sin_t = sin(theta);
    
    // Construct axes
    vec2 axis1 = vec2(cos_t, sin_t) * r1;
    vec2 axis2 = vec2(-sin_t, cos_t) * r2;
    
    // Vertex position in screen space (pixels? no, NDC usually)
    // But J was in camera space units? 
    // Actually J maps to image plane.
    
    // Let's simplify. If we use the standard "Billboard with 2D Covariance" approach:
    // We calculate the 2D offset in screen space.
    
    vec2 offset = quad_pos.x * axis1 + quad_pos.y * axis2;
    
    // Apply offset to projected center
    vec4 center_clip = projectionMatrix * pos_cam;
    vec2 center_ndc = center_clip.xy / center_clip.w;
    
    // We need to scale offset by viewport size to get NDC?
    // J * T * J' gives covariance in... what units?
    // If J uses focal length in pixels, then V2d is in pixels^2.
    // Then offset is in pixels.
    // Convert pixels to NDC: offset / viewportSize * 2.0
    
    vec2 offset_ndc = (offset / viewportSize) * 2.0;
    
    gl_Position = vec4(center_ndc + offset_ndc, center_clip.z / center_clip.w, 1.0);
}
"""

# Fallback Simple Shader (Billboard) - Safest for first iteration
VERT_SHADER_SIMPLE = """
uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform vec3 camPos;

in vec3 pos;
in vec4 color;
in vec3 scale;
in vec4 rot;
in vec2 quad_pos;

out vec4 vColor;
out vec2 vUv;

void main() {
    vColor = color;
    vUv = quad_pos;

    // Simple Billboard: Face Camera
    vec3 forward = normalize(camPos - pos);
    vec3 right = normalize(cross(vec3(0.0, 0.0, 1.0), forward));
    vec3 up = cross(forward, right);
    
    // Use average scale
    float s = (scale.x + scale.y + scale.z) / 3.0;
    
    vec3 world_pos = pos + (right * quad_pos.x * s) + (up * quad_pos.y * s);
    gl_Position = viewProjectionMatrix * vec4(world_pos, 1.0);
}
"""

FRAG_SHADER = """
in vec4 vColor;
in vec2 vUv;
out vec4 fragColor;

void main() {
    // Gaussian falloff
    float r2 = dot(vUv, vUv);
    if (r2 > 1.0) discard;
    
    float alpha = exp(-0.5 * r2 * 9.0); // 3 sigma?
    // Or standard Gaussian: exp(-dist^2)
    // vUv is -1 to 1.
    
    // Let's use simple alpha falloff
    float a = exp(-2.0 * r2);
    
    fragColor = vec4(vColor.rgb, vColor.a * a);
}
"""

class LCCGLSLRenderer:
    def __init__(self):
        self.shader = None
        self.batch = None
        self.num_instances = 0
        self._setup_shader()
        
    def _setup_shader(self):
        # Using the simple shader for now to ensure we get something on screen.
        # We can upgrade to the covariance one later.
        self.shader = gpu.types.GPUShader(VERT_SHADER_SIMPLE, FRAG_SHADER)
        
    def load_data(self, pos, col, scl, rot):
        """
        pos: (N, 3) float32
        col: (N, 4) float32 (0-1)
        scl: (N, 3) float32
        rot: (N, 4) float32 (Quaternion)
        """
        self.num_instances = len(pos)
        if self.num_instances == 0:
            self.batch = None
            return

        # Store original data for sorting
        self.original_pos = pos
        self.original_col = col
        self.original_scl = scl
        self.original_rot = rot
        
        # Quad Vertices (shared)
        quad_verts = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0]
        ], dtype=np.float32)
        
        # Indices for two triangles
        indices = [(0, 1, 2), (0, 2, 3)]
        
        # 1. Quad Buffer (Per Vertex)
        fmt_quad = gpu.types.GPUVertFormat()
        fmt_quad.attr_add(id="quad_pos", comp_type='F32', len=2, fetch_mode='FLOAT')
        vbo_quad = gpu.types.GPUVertBuf(len=quad_verts, format=fmt_quad)
        vbo_quad.attr_fill(id="quad_pos", data=quad_verts)
        
        # 2. Instance Buffer (Per Instance)
        fmt_inst = gpu.types.GPUVertFormat()
        fmt_inst.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt_inst.attr_add(id="color", comp_type='F32', len=4, fetch_mode='FLOAT')
        fmt_inst.attr_add(id="scale", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt_inst.attr_add(id="rot", comp_type='F32', len=4, fetch_mode='FLOAT')
        
        self.vbo_inst = gpu.types.GPUVertBuf(len=pos, format=fmt_inst)
        self.vbo_inst.attr_fill(id="pos", data=pos)
        self.vbo_inst.attr_fill(id="color", data=col)
        self.vbo_inst.attr_fill(id="scale", data=scl)
        self.vbo_inst.attr_fill(id="rot", data=rot)
        
        # 3. Create Batch
        idx_data = []
        for tri in indices:
            idx_data.extend(tri)
        ibo = gpu.types.GPUIndexBuf(type='TRIANGLES', seq=idx_data)
        
        self.batch = gpu.types.GPUBatch(type='TRIANGLES', buf=vbo_quad, elem=ibo)
        self.batch.inst_add(self.vbo_inst)
        
    def sort_and_update(self, cam_pos):
        if self.num_instances == 0 or not hasattr(self, 'original_pos'):
            return
            
        # Calculate squared distances
        # cam_pos is Vector, convert to numpy
        cam_p = np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32)
        diff = self.original_pos - cam_p
        dists = np.einsum('ij,ij->i', diff, diff) # Faster squared norm
        
        # Sort indices (descending for back-to-front)
        indices = np.argsort(dists)[::-1]
        
        # Re-order data
        # Note: This is CPU intensive for large N.
        # Optimization: Only sort every N frames or if camera moved significantly.
        pos_sorted = self.original_pos[indices]
        col_sorted = self.original_col[indices]
        scl_sorted = self.original_scl[indices]
        rot_sorted = self.original_rot[indices]
        
        # Update VBO
        self.vbo_inst.attr_fill(id="pos", data=pos_sorted)
        self.vbo_inst.attr_fill(id="color", data=col_sorted)
        self.vbo_inst.attr_fill(id="scale", data=scl_sorted)
        self.vbo_inst.attr_fill(id="rot", data=rot_sorted)

    def draw(self):
        if not self.batch or not self.shader:
            return
            
        self.shader.bind()
        
        # Set Uniforms
        matrix = bpy.context.region_data.perspective_matrix
        view_matrix = bpy.context.region_data.view_matrix
        
        # Get Camera Position from View Matrix (more reliable for viewport)
        # View Matrix is World -> Camera. Inverse is Camera -> World.
        # Translation of inverse is camera position.
        cam_pos = view_matrix.inverted().translation
        
        self.shader.uniform_float("viewProjectionMatrix", matrix)
        self.shader.uniform_float("viewMatrix", view_matrix)
        self.shader.uniform_float("camPos", cam_pos)
        
        # Pass viewport size for 2D projection logic if needed
        # viewport_size = Vector((bpy.context.region.width, bpy.context.region.height))
        # self.shader.uniform_float("viewportSize", viewport_size)
        
        self.batch.draw_instanced(self.shader)

