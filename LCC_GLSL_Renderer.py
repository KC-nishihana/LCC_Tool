import bpy
import gpu
from gpu_extras.batch import batch_for_shader
import numpy as np
from mathutils import Matrix, Vector

# --------------------------------------------------------------------
# GLSL Shaders (3DGS EWA Splatting) - Texture Based Instancing
# --------------------------------------------------------------------

VERT_SHADER = """
uniform mat4 viewProjectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform vec2 viewportSize;
uniform vec2 focal; // x, y

uniform sampler2D posTex;
uniform sampler2D colorTex;
uniform sampler2D scaleTex;
uniform sampler2D rotTex;
uniform int texWidth;

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
    // Fetch Instance Data from Textures
    int id = gl_InstanceID;
    int y = id / texWidth;
    int x = id % texWidth;
    ivec2 texCoord = ivec2(x, y);

    vec3 pos = texelFetch(posTex, texCoord, 0).xyz;
    vec4 color = texelFetch(colorTex, texCoord, 0);
    vec3 scale = texelFetch(scaleTex, texCoord, 0).xyz;
    vec4 rot = texelFetch(rotTex, texCoord, 0);

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
    float x_cam = pos_cam.x;
    float y_cam = pos_cam.y;
    
    mat3 J = mat3(
        focal.x / z, 0.0, -(focal.x * x_cam) / (z * z),
        0.0, focal.y / z, -(focal.y * y_cam) / (z * z),
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
        self.batch = None
        self.num_instances = 0
        
        # Textures
        self.pos_tex = None
        self.col_tex = None
        self.scl_tex = None
        self.rot_tex = None
        self.tex_width = 2048 # Fixed width for textures
        
        self._setup_shader()
        
    def _setup_shader(self):
        try:
            self.shader = gpu.types.GPUShader(VERT_SHADER, FRAG_SHADER)
        except Exception as e:
            print(f"Shader Compilation Error: {e}")
            self.shader = None
        
    def _create_data_texture(self, data, width):
        """
        Creates a GPUTexture from numpy data.
        Data must be float32.
        Pads data to fit width * height.
        Returns texture object.
        """
        num_items = len(data)
        num_components = data.shape[1]
        
        height = (num_items + width - 1) // width
        padded_size = width * height
        
        # Pad data
        if num_items < padded_size:
            padding = np.zeros((padded_size - num_items, num_components), dtype=np.float32)
            data_padded = np.vstack([data, padding])
        else:
            data_padded = data
            
        # Ensure 4 components for RGBA32F
        if num_components == 3:
            # Add alpha channel (1.0 or 0.0 doesn't matter much for pos/scale, but 0 is safer)
            alpha = np.zeros((padded_size, 1), dtype=np.float32)
            data_final = np.hstack([data_padded, alpha])
        else:
            data_final = data_padded
            
        # Flatten
        data_flat = data_final.flatten()
        
        # Create Buffer
        # gpu.types.Buffer(type, size, data)
        # type: 'FLOAT', 'INT', 'UINT', 'UCHAR'
        # size: number of elements (not bytes)
        gpu_buffer = gpu.types.Buffer('FLOAT', len(data_flat), data_flat)
        
        # Create Texture
        tex = gpu.types.GPUTexture((width, height), format='RGBA32F', data=gpu_buffer)
        return tex

    def load_data(self, pos, col, scl, rot):
        """
        Loads data into GPU Textures.
        Expects numpy arrays.
        """
        self.num_instances = len(pos)
        if self.num_instances == 0:
            self.batch = None
            return

        # Keep references for sorting (optional, not implemented for texture yet)
        self.original_pos = pos
        self.original_col = col
        self.original_scl = scl
        self.original_rot = rot
        
        # Create Textures
        try:
            self.pos_tex = self._create_data_texture(pos, self.tex_width)
            self.col_tex = self._create_data_texture(col, self.tex_width)
            self.scl_tex = self._create_data_texture(scl, self.tex_width)
            self.rot_tex = self._create_data_texture(rot, self.tex_width)
            print(f"Created Data Textures: {self.pos_tex.width}x{self.pos_tex.height} for {self.num_instances} instances.")
        except Exception as e:
            print(f"Failed to create data textures: {e}")
            import traceback
            traceback.print_exc()
            return

        # Create Batch (Quad Only)
        quad_verts = np.array([
            [-1.0, -1.0], [ 1.0, -1.0],
            [ 1.0,  1.0], [-1.0,  1.0]
        ], dtype=np.float32)
        
        fmt_quad = gpu.types.GPUVertFormat()
        fmt_quad.attr_add(id="quad_pos", comp_type='F32', len=2, fetch_mode='FLOAT')
        vbo_quad = gpu.types.GPUVertBuf(len=len(quad_verts), format=fmt_quad)
        vbo_quad.attr_fill(id="quad_pos", data=quad_verts)
        
        indices = [(0, 1, 2), (0, 2, 3)]
        ibo = gpu.types.GPUIndexBuf(type='TRIS', seq=indices)
        
        try:
            self.batch = gpu.types.GPUBatch(type='TRIS', buf=vbo_quad, elem=ibo)
            print("DEBUG: GPUBatch created successfully (Quad Only).")
        except Exception as e:
            print(f"DEBUG: Failed to create GPUBatch: {e}")
            self.batch = None
    
    def sort_and_update(self, cam_pos):
        """Sort instances back-to-front based on camera position"""
        # Sorting with textures requires re-uploading the textures.
        # This can be expensive. Disabled for now.
        pass

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
        self.shader.uniform_int("texWidth", self.tex_width)
        
        # Bind Textures
        if self.pos_tex: self.shader.uniform_sampler("posTex", self.pos_tex)
        if self.colorTex: self.shader.uniform_sampler("colorTex", self.col_tex)
        if self.scaleTex: self.shader.uniform_sampler("scaleTex", self.scl_tex)
        if self.rotTex: self.shader.uniform_sampler("rotTex", self.rot_tex)
        
        # Draw
        self.batch.draw_instanced(self.shader)

    # Helper properties for uniform binding names (avoiding typos)
    @property
    def colorTex(self): return self.col_tex
    @property
    def scaleTex(self): return self.scl_tex
    @property
    def rotTex(self): return self.rot_tex

