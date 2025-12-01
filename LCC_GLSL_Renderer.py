import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from mathutils import Matrix, Vector

# --------------------------------------------------------------------
# GLSL Shaders (3DGS EWA Splatting)
# --------------------------------------------------------------------

VERT_SHADER = """
uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 viewportSize;
uniform vec2 focal; // x, y

in vec3 pos;          // Instance Attribute: Center
in vec4 color;        // Instance Attribute: Color
in vec3 scale;        // Instance Attribute: Scale
in vec4 rot;          // Instance Attribute: Rotation (Quaternion)
in vec2 quad_pos;     // Vertex Attribute: Quad Coord

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

    // Transform center to camera space
    vec4 pos_cam = viewMatrix * vec4(pos, 1.0);
    
    // Simple culling (behind camera)
    if (pos_cam.z > -0.2) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0); // Clip
        return;
    }

    // 1. Compute 3D Covariance Matrix
    mat3 R = quatToMat3(rot);
    mat3 S = mat3(
        scale.x, 0.0, 0.0,
        0.0, scale.y, 0.0,
        0.0, 0.0, scale.z
    );
    mat3 M = R * S;
    mat3 V = M * transpose(M); // Covariance in World Space
    
    // 2. Project to 2D
    // Jacobian of the perspective projection
    // J = [ fx/z  0   -fx*x/z^2 ]
    //     [ 0   fy/z  -fy*y/z^2 ]
    
    // Note: Blender Camera Space looks down -Z.
    // pos_cam.z is negative.
    
    float z = pos_cam.z;
    float x = pos_cam.x;
    float y = pos_cam.y;
    
    mat3 J = mat3(
        focal.x / z, 0.0, -(focal.x * x) / (z * z),
        0.0, focal.y / z, -(focal.y * y) / (z * z),
        0.0, 0.0, 0.0
    );
    
    // Transform Covariance to Camera Space then Ray Space
    // W is the rotation part of ViewMatrix
    mat3 W = mat3(viewMatrix);
    mat3 T = W * V * transpose(W); // Covariance in Camera Space
    
    // 2D Covariance in Screen Space
    mat3 V2d = J * T * transpose(J);
    
    // 2x2 Covariance elements
    float cov00 = V2d[0][0];
    float cov01 = V2d[0][1];
    float cov11 = V2d[1][1];
    
    // Add low-pass filter to prevent aliasing (approx pixel size)
    cov00 += 0.3;
    cov11 += 0.3;

    // Eigendecomposition to find axes of the ellipse
    float lambda1 = 0.5 * (cov00 + cov11 + sqrt(max(0.0, (cov00 - cov11) * (cov00 - cov11) + 4.0 * cov01 * cov01)));
    float lambda2 = 0.5 * (cov00 + cov11 - sqrt(max(0.0, (cov00 - cov11) * (cov00 - cov11) + 4.0 * cov01 * cov01)));
    
    // Radius (3 sigma covers 99%)
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
    
    // Vertex offset in Screen Pixels
    vec2 offset = quad_pos.x * axis1 + quad_pos.y * axis2;
    
    // Convert Offset to NDC
    vec2 offset_ndc = (offset / viewportSize) * 2.0;
    
    // Projected center in NDC
    vec4 center_clip = projectionMatrix * pos_cam;
    vec2 center_ndc = center_clip.xy / center_clip.w;
    
    gl_Position = vec4(center_ndc + offset_ndc, center_clip.z / center_clip.w, 1.0);
}
"""

FRAG_SHADER = """
in vec4 vColor;
in vec2 vUv;
out vec4 fragColor;

void main() {
    // Gaussian falloff calculation
    // vUv is in range [-1, 1]
    float r2 = dot(vUv, vUv);
    
    if (r2 > 1.0) discard;
    
    // Gaussian function: exp(-0.5 * x^2 / sigma^2)
    // Here we map radius 1 to 3 sigma.
    float alpha_falloff = exp(-0.5 * r2 * 9.0); // 3^2 = 9
    
    // Simple alpha composition approximation
    float final_alpha = vColor.a * alpha_falloff;
    
    if (final_alpha < 0.01) discard;
    
    fragColor = vec4(vColor.rgb, final_alpha);
}
"""

class LCCGLSLRenderer:
    def __init__(self):
        self.shader = None
        self.dummy_shader = None # Helper for batch creation
        self.batch = None
        self.num_instances = 0
        self._setup_shader()
        
    def _setup_shader(self):
        try:
            self.shader = gpu.types.GPUShader(VERT_SHADER, FRAG_SHADER)
            
            # Dummy shader for batch creation (only knows about quad_pos)
            dummy_vert = """
            in vec2 quad_pos;
            void main() { gl_Position = vec4(quad_pos, 0.0, 1.0); }
            """
            dummy_frag = """
            out vec4 fragColor;
            void main() { fragColor = vec4(1.0); }
            """
            self.dummy_shader = gpu.types.GPUShader(dummy_vert, dummy_frag)
            
        except Exception as e:
            print(f"Shader Compilation Error: {e}")
            self.shader = None
            self.dummy_shader = None
        
    def load_data(self, pos, col, scl, rot):
        """
        Loads data into GPU buffers.
        Expects numpy arrays.
        """
        self.num_instances = len(pos)
        if self.num_instances == 0:
            self.batch = None
            return

        # Keep references for sorting
        self.original_pos = pos
        self.original_col = col
        self.original_scl = scl
        self.original_rot = rot
        
        # Base Quad (Per Vertex)
        quad_verts = np.array([
            [-1.0, -1.0], [ 1.0, -1.0],
            [ 1.0,  1.0], [-1.0,  1.0]
        ], dtype=np.float32)
        
        # 1. Create Quad VBO manually
        fmt_quad = gpu.types.GPUVertFormat()
        fmt_quad.attr_add(id="quad_pos", comp_type='F32', len=2, fetch_mode='FLOAT')
        vbo_quad = gpu.types.GPUVertBuf(len=len(quad_verts), format=fmt_quad)
        vbo_quad.attr_fill(id="quad_pos", data=quad_verts)

        # 2. Create Instance VBO manually
        fmt_inst = gpu.types.GPUVertFormat()
        fmt_inst.attr_add(id="pos", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt_inst.attr_add(id="color", comp_type='F32', len=4, fetch_mode='FLOAT')
        fmt_inst.attr_add(id="scale", comp_type='F32', len=3, fetch_mode='FLOAT')
        fmt_inst.attr_add(id="rot", comp_type='F32', len=4, fetch_mode='FLOAT')
        self.vbo_inst = gpu.types.GPUVertBuf(len=len(pos), format=fmt_inst)
        self.update_buffers(pos, col, scl, rot)

        # 3. Create IBO manually (using list of tuples)
        # It seems GPUIndexBuf expects the structure matching the type (TRIS -> list of 3-tuples)
        try:
            ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=indices)
            print("DEBUG: GPUIndexBuf created successfully with list of tuples.")
        except Exception as e:
            print(f"DEBUG: Failed to create GPUIndexBuf with tuples: {e}")
            # Last ditch attempt: flat list? No, we tried that.
            # What if we just don't use an IBO? (Not possible for Quad instancing usually unless we duplicate verts)
            ibo = None

        if ibo:
            # 4. Create Final Batch with BOTH buffers
            try:
                self.batch = gpu.types.GPUBatch(type='TRIS', buf=[vbo_quad, self.vbo_inst], elem=ibo)
                print("DEBUG: GPUBatch created successfully with VBO list.")
            except Exception as e:
                print(f"DEBUG: Failed to create GPUBatch with list: {e}")
                self.batch = None
        else:
            print("Error: Could not create IBO.")
    
    def update_buffers(self, pos, col, scl, rot):
        """Update instance VBO with new (sorted) data"""
        self.vbo_inst.attr_fill(id="pos", data=pos)
        self.vbo_inst.attr_fill(id="color", data=col)
        self.vbo_inst.attr_fill(id="scale", data=scl)
        self.vbo_inst.attr_fill(id="rot", data=rot)

    def sort_and_update(self, cam_pos):
        """Sort instances back-to-front based on camera position"""
        if self.num_instances == 0 or not hasattr(self, 'original_pos'):
            return
            
        # 1. Calculate Squared Distances
        # Note: In Blender, camera looks down -Z. 
        # But for sorting simple euclidean distance is usually fine for splats.
        # For strict 3DGS, we project to view vector, but distance is a good approximation.
        
        # Convert cam_pos to numpy
        cam_p = np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32)
        
        # Vector from camera to splat
        diff = self.original_pos - cam_p
        
        # Squared distance (faster than norm)
        dists = np.einsum('ij,ij->i', diff, diff)
        
        # 2. Argsort (Back-to-Front means furthest first -> Descending order)
        # numpy argsort is ascending, so we take [::-1]
        indices = np.argsort(dists)[::-1]
        
        # 3. Reorder Data
        pos_sorted = self.original_pos[indices]
        col_sorted = self.original_col[indices]
        scl_sorted = self.original_scl[indices]
        rot_sorted = self.original_rot[indices]
        
        # 4. Upload to GPU
        self.update_buffers(pos_sorted, col_sorted, scl_sorted, rot_sorted)

    def draw(self):
        if not self.batch or not self.shader:
            return
            
        self.shader.bind()
        
        # Context Data
        region = bpy.context.region
        region_data = bpy.context.region_data
        
        if not region_data:
            return

        # Matrices
        view_matrix = region_data.view_matrix
        projection_matrix = region_data.window_matrix
        
        # Calculate Focal Length (fx, fy) from Projection Matrix
        # OpenGL Projection: P[0][0] = 2n / (r-l) = 2 * focal_x / width
        # focal_x = P[0][0] * width / 2
        width = region.width
        height = region.height
        
        fx = projection_matrix[0][0] * width / 2.0
        fy = projection_matrix[1][1] * height / 2.0
        
        # Uniforms
        self.shader.uniform_float("viewProjectionMatrix", projection_matrix @ view_matrix)
        self.shader.uniform_float("viewMatrix", view_matrix)
        self.shader.uniform_float("projectionMatrix", projection_matrix)
        self.shader.uniform_float("viewportSize", Vector((width, height)))
        self.shader.uniform_float("focal", Vector((fx, fy)))
        
        # Draw
        self.batch.draw_instanced(self.shader)
