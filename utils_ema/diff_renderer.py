import numpy as np
import nvdiffrast.torch as dr
import torch
from utils_ema.image import Image
from utils_ema.texture import Texture
from utils_ema.torch_utils import get_device
import os

# os.environ['TORCH_CUDA_ARCH_LIST']


class Renderer:
    """Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.

    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    glctx = dr.RasterizeGLContext()
    near = 0.001
    far = 1e20

    @classmethod
    def set_near_far(cls, cams, samples, epsilon=0.00001):
        """Automatically adjust the near and far plane distance"""

        device = cams[0][0].device

        mins = []
        maxs = []
        for frame in cams:
            for cam in frame:
                samples_projected = cam.project(
                    samples.to(device), depth_as_distance=True
                )
                mins.append(samples_projected[..., 2].min())
                maxs.append(samples_projected[..., 2].max())

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
    def projection(fx, fy, cx, cy, n, f, width, height, device, dtype):
        # print(fx.device)
        # print(fy.device)
        # print(cx.device)
        # print(cy.device)
        # print(width.device)
        # print(height.device)
        # print(n.device)
        # print(f.device)
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        """
        return torch.tensor(
            [
                [2.0 * fx / width, 0, 1.0 - 2.0 * cx / width, 0],
                [0, 2.0 * fy / height, 1.0 - 2.0 * cy / height, 0],
                [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                [0, 0, -1, 0.0],
            ],
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def to_gl_camera(camera, resolution, n=1000, f=5000):

        # import ipdb; ipdb.set_trace()

        projection_matrix = Renderer.projection(
            fx=camera.K[0, 0],
            fy=camera.K[1, 1],
            cx=camera.K[0, 2],
            cy=camera.K[1, 2],
            n=n,
            f=f,
            width=resolution[1],
            height=resolution[0],
            device=camera.device,
            dtype=camera.dtype,
        )

        # projection_matrix = Renderer.projection(fx=camera.K[1,1],
        #                                         fy=camera.K[0,0],
        #                                         cx=camera.K[1,2],
        #                                         cy=camera.K[0,2],
        #                                         n=n,
        #                                         f=f,
        #                                         width=resolution[0],
        #                                         height=resolution[1],
        #                                         device=camera.device)

        Rt = torch.eye(4, device=camera.device, dtype=projection_matrix.dtype)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t

        gl_transform = torch.tensor(
            [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]],
            device=camera.device,
            dtype=Rt.dtype,
        )

        Rt = gl_transform @ Rt
        return projection_matrix @ Rt

    # @classmethod
    # def render_neural(cls, camera, obj, neural_shader):
    #     gbuffers = cls.diffrast(camera, obj, channels=['mask', 'position', 'normal'], with_antialiasing=True)
    #     mask = (gbuffers["mask"] > 0).squeeze()
    #     # indexes = torch.nonzero(mask).squeeze()

    #     pixs = camera.sample_rand_pixs_in_mask( mask, percentage=1)
    #     dirs = camera.pix2dir( pixs )

    #     if pixs is None: return None

    #     # images
    #     position = gbuffers["position"]
    #     normal = gbuffers["normal"]
    #     img = torch.zeros( normal.shape , device=position.device)

    #     position = position[pixs[:, 1], pixs[:, 0], :]
    #     normal = normal[pixs[:, 1], pixs[:, 0], :]
    #     shaded = neural_shader(position, normal, dirs)

    #     img[pixs[:,1],pixs[:,0],:]=shaded
    #     return Image(img=img)

    @classmethod
    def diffrast(
        cls, camera, obj, channels, with_antialiasing=False, get_rast_idx=False
    ):

        # import ipdb
        #
        # ipdb.set_trace()

        device = get_device()

        # camera.dtype(torch.float64)
        # obj.mesh.type(torch.float64)
        # obj.pose.dtype(torch.float64)
        #
        assert device != "cpu"

        gbuffer = {}

        gl_cam = camera.get_camera_opencv()
        gl_cam.type(obj.mesh.uv.dtype)
        r = camera.intr.resolution
        r = [r[1], r[0]]
        P = Renderer.to_gl_camera(gl_cam, r, n=cls.near, f=cls.far)
        uv = obj.mesh.uv.to(device)

        # v = obj.mesh.vertices.to(device)
        # n = obj.mesh.vertex_normals.to(device)
        # idx = obj.mesh.indices.int().to(device)
        # R = obj.pose.rotation().to(device)
        # # pos = Renderer.transform_pos(P, ( v@R.t() )+l)
        # pos = Renderer.transform_pos(P, v+l)

        # import ipdb; ipdb.set_trace()

        # v = obj.mesh.vertices+l
        # n = obj.mesh.vertex_normals.to(device)
        # pos = Renderer.transform_pos(P, v)
        # idx = obj.mesh.indices.int().to(device)

        l = obj.pose.location().to(device)
        R = obj.pose.rotation().to(device)
        s = obj.pose.scale.to(device)
        v_mesh = obj.mesh.vertices.to(device)
        v = ((s * v_mesh) @ R.t()) + l

        # v = obj.get_vertices_from_pose()

        mesh = obj.mesh.with_vertices(v)
        # v = mesh.vertices.to(device)
        n = mesh.vertex_normals.to(device)
        idx = mesh.indices.int().to(device)
        pos = Renderer.transform_pos(P, v)

        # pos = pos.type(torch.float32)
        # camera.dtype(torch.float32)
        # obj.mesh.type(torch.float32)
        # obj.pose.dtype(torch.float32)
        #
        # import ipdb
        #
        # ipdb.set_trace()

        rast, _ = dr.rasterize(cls.glctx, pos, idx, resolution=r)

        # Collect arbitrary output variables (aovs)
        if "mask" in channels:
            mask = torch.clamp(rast[..., -1:], 0, 1)
            gbuffer["mask"] = (
                dr.antialias(mask, rast, pos, idx)[0] if with_antialiasing else mask[0]
            )
        else:
            mask = torch.clamp(rast.detach()[..., -1:], 0, 1)
            gbuffer["mask"] = mask[0]

        if "position" in channels or "depth" in channels:
            # position, _ = dr.interpolate(mesh.vertices[None, ...], rast, idx)
            position, _ = dr.interpolate(v[None, ...], rast, idx)
            # gbuffer["position"] = gbuffer["mask"]
            gbuffer["position"] = (
                dr.antialias(position, rast, pos, idx)[0]
                if with_antialiasing
                else position[0]
            )

        if "normal" in channels:
            # normal, _ = dr.interpolate(mesh.vertex_normals[None, ...], rast, idx)
            normal, _ = dr.interpolate(n[None, ...], rast, idx)
            # gbuffer["normal"] = gbuffer["position"]
            gbuffer["normal"] = (
                dr.antialias(normal, rast, pos, idx)[0]
                if with_antialiasing
                else normal[0]
            )

        if "depth" in channels:
            gbuffer["depth"] = camera.project(
                gbuffer["position"], depth_as_distance=True
            )[..., 2:3]

        if "uv" in channels:
            uv, _ = gbuffer["uv"] = dr.interpolate(uv[None, ...], rast, idx)
            gbuffer["uv"] = (
                dr.antialias(uv, rast, pos, idx)[0] if with_antialiasing else uv[0]
            )

        # del pos, idx
        # torch.cuda.empty_cache()
        if get_rast_idx:
            return gbuffer, rast, idx

        return gbuffer

    @classmethod
    # def get_buffers_pixels_dirs(cls, camera, obj, shading_percentage=1, channels=['mask', 'position', 'normal'], no_contour=True, with_antialiasing=False):
    def get_buffers_pixels_dirs(
        cls,
        camera,
        obj,
        n_pixs=None,
        channels=["mask", "position", "normal"],
        no_contour=True,
        with_antialiasing=False,
    ):

        if "mask" not in channels:
            channels += "mask"

        gbuffers = Renderer.diffrast(
            camera, obj, channels=channels, with_antialiasing=with_antialiasing
        )
        # Image(gbuffers["uv"][:, :, 0]).show()

        if no_contour:
            gbuffers["mask"][0, :] = 0
            gbuffers["mask"][:, 0] = 0
            gbuffers["mask"][-1, :] = 0
            gbuffers["mask"][:, -1] = 0

        # sample pixels in mask
        mask = (gbuffers["mask"] > 0).squeeze()

        # pixs = camera.sample_rand_pixs_in_mask( mask, percentage=shading_percentage)
        if n_pixs is None:
            pixs = torch.nonzero(mask, as_tuple=False).to(mask.device)
        else:
            pixs = camera.sample_rand_pixs_in_mask(mask, n_pixs=n_pixs)
        if pixs is None:
            return None, None, None
        dirs = camera.pix2dir(pixs).to(mask.device)

        if "mask" not in channels:
            del mask
            torch.cuda.empty_cache()

        return gbuffers, pixs, dirs

    @classmethod
    def render_texture(cls, camera, obj, texture, with_antialiasing=False):
        gbuffers, pixs, _ = cls.get_buffers_pixels_dirs(
            camera, obj, channels=["mask", "uv"], with_antialiasing=with_antialiasing
        )
        uv = gbuffers["uv"]
        image = torch.zeros(list(uv.shape[:-1]) + [3], device=uv.device)

        texture = texture.to(uv.device)
        texture.img = texture.img.flip(0)
        res = texture.res

        pix_tex = (uv[pixs[:, 1], pixs[:, 0], :] * res).long()
        image[pixs[:, 1], pixs[:, 0], :] = texture.img[pix_tex[:, 1], pix_tex[:, 0], :]
        return Image(image), pixs
