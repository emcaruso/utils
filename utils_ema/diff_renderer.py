import numpy as np
import nvdiffrast.torch as dr
import torch


class Renderer:
    """ Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.

    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    glctx = dr.RasterizeGLContext()
    near = 1
    far = 1000


    @classmethod
    def set_near_far(cls, cams, samples, epsilon=0.1):
        """ Automatically adjust the near and far plane distance
        """

        mins = []
        maxs = []
        for frame in cams:
            for cam in frame:
                samples_projected = cam.project(samples, depth_as_distance=True)
                mins.append(samples_projected[...,2].min())
                maxs.append(samples_projected[...,2].max())

        near, far = min(mins), max(maxs)
        cls.near = near - (near * epsilon)
        cls.far = far + (far * epsilon)

    @staticmethod
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[:, 0:1])], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    @staticmethod
    def projection(fx, fy, cx, cy, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        """
        return torch.tensor([[2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0],
                            [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], device=device) 
    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000):

        projection_matrix = Renderer.projection(fx=camera.K[0,0],
                                                fy=camera.K[1,1],
                                                cx=camera.K[0,2],
                                                cy=camera.K[1,2],
                                                n=n,
                                                f=f,
                                                width=resolution[1],
                                                height=resolution[0],
                                                device=camera.device)

        Rt = torch.eye(4, device=camera.device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        gl_transform = torch.tensor([[1., 0,  0,  0],
                                    [0,  1., 0,  0],
                                    [0,  0, -1., 0],
                                    [0,  0,  0,  1.]], device=camera.device)

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt

    @classmethod
    def render(cls, camera, obj, channels, with_antialiasing=True):
        """ Render G-buffers from a set of views.

        Args:
            views (List[Views]): 
        """

        # TODO near far should be passed by view to get higher resolution in depth
        gbuffer = {}


        gl_cam = camera.get_camera_opencv()
        # print(gl_cam.K)
        # print(gl_cam.R)
        # print(gl_cam.t)
        # # print(gl_cam.resolution)
        # # print(self.near)
        # # print(self.far)
        # exit(1)
        # Rasterize only once
        P = Renderer.to_gl_camera( gl_cam, camera.intr.resolution, n=cls.near, f=cls.far)
        pos = Renderer.transform_pos(P, obj["mesh"].vertices+obj["pose"].location())
        idx = obj["mesh"].indices.int()
        rast, rast_out_db = dr.rasterize(cls.glctx, pos, idx, resolution=camera.intr.resolution)

        # Collect arbitrary output variables (aovs)
        if "mask" in channels:
            mask = torch.clamp(rast[..., -1:], 0, 1)
            gbuffer["mask"] = dr.antialias(mask, rast, pos, idx)[0] if with_antialiasing else mask[0]

        if "position" in channels or "depth" in channels:
            position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)
            gbuffer["position"] = dr.antialias(position, rast, pos, idx)[0] if with_antialiasing else position[0]

        if "normal" in channels:
            normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)
            gbuffer["normal"] = dr.antialias(normal, rast, pos, idx)[0] if with_antialiasing else normal[0]

        if "depth" in channels:
            gbuffer["depth"] = camera.project(gbuffer["position"], depth_as_distance=True)[..., 2:3]

        return gbuffer
    
