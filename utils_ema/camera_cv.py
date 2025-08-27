import cv2
from typing import Optional, Tuple
import pyrender
import torch
import sys, os
import copy as cp
from typing import Union
from utils_ema.geometry_pose import *
from utils_ema.plot import *
from utils_ema.torch_utils import *
from utils_ema.general import *
from utils_ema.image import *


class Camera_opencv:
    """Camera in OpenCV format.

    Args:
        K (tensor): Camera matrix with intrinsic parameters (3x3)
        R (tensor): Rotation matrix (3x3)
        t (tensor): translation vector (3)
        device (torch.device): Device where the matrices are stored
    """

    def __init__(self, K, R, t, device="cpu", dtype=torch.float32):
        self.K = K.to(device) if torch.is_tensor(K) else torch.FloatTensor(K).to(device)
        self.R = R.to(device) if torch.is_tensor(R) else torch.FloatTensor(R).to(device)
        self.t = t.to(device) if torch.is_tensor(t) else torch.FloatTensor(t).to(device)
        self.device = device
        self.dtype = dtype

    def to(self, device="cpu"):
        self.K = self.K.to(device)
        self.R = self.R.to(device)
        self.t = self.t.to(device)
        self.device = device
        return self

    def type(self, dtype):
        self.K.type(dtype)
        self.R.type(dtype)
        self.t.type(dtype)
        self.dtype = dtype

    @property
    def center(self):
        return -self.R.t() @ self.t

    @property
    def P(self):
        return self.K @ torch.cat([self.R, self.t.unsqueeze(-1)], dim=-1)


class Intrinsics:
    def __init__(
        self,
        K: torch.Tensor = torch.FloatTensor(
            [[0.016, 0, 0.006], [0, 0.016, 0.006], [0, 0, 1]]
        ),
        D: Union[torch.Tensor, None] = None,
        resolution: torch.Tensor = torch.LongTensor([30, 20]),
        sensor_size: torch.Tensor = torch.FloatTensor([0.012, 0.012]),
        units: str = "meters",
        dtype=torch.float32,
        device: str = "cpu",
    ):
        self.units = units
        self.dtype = dtype
        self.sensor_size = sensor_size
        self.resolution = resolution
        self.device = device
        self.D_params = D if D is not None else None
        self.K_params = torch.zeros(K.shape[:-2] + (4,), device=self.device)
        self.K_params[..., 0] = K[..., 0, 0]
        self.K_params[..., 1] = K[..., 1, 1]
        self.K_params[..., 2] = K[..., 0, 2]
        self.K_params[..., 3] = K[..., 1, 2]
        self.update_intrinsics()
        self.type(dtype)
        self.compute_undistortion_map()

    def update_intrinsics(self):
        self.K = self.get_K()
        self.K_pix = self.get_K_pix()
        self.K_und, self.K_pix_und, self.roi_und = self.get_K_und()

    # Get K from params (differentiable)
    def get_K(self):
        fx = self.K_params[..., 0]
        fy = self.K_params[..., 1]
        cx = self.K_params[..., 2]
        cy = self.K_params[..., 3]
        zero = torch.zeros_like(fx, device=self.device)
        K = torch.stack(
            [
                torch.stack([fx, zero, cx], dim=-1),
                torch.stack([zero, fy, cy], dim=-1),
                torch.tensor([0, 0, 1], device=self.device, dtype=self.dtype).expand(
                    *fx.shape, 3
                ),
            ],
            dim=-2,
        )
        return K

    # Get K from params (differentiable)
    def get_K_pix(self):
        r = self.pixel_unit_ratio()
        K_pix = self.get_K() * r.unsqueeze(-1)
        K_pix[..., 2, 2] = 1
        return K_pix

    # Get K_und and K_pix_und from params (NON differentiable)
    def get_K_und(self, alpha=0, central_pp=False, same_fx_fy=True):
        K_pix_und = None
        K_und = None
        roi_und = None
        if self.D_params is not None:
            w = self.resolution[..., 0].type(torch.int32)
            h = self.resolution[..., 1].type(torch.int32)

            # K_pix_und, roi_und = cv2.getOptimalNewCameraMatrix(
            #     self.get_K_pix().cpu().numpy(),
            #     self.D_params.cpu().numpy(),
            #     (w, h),
            #     alpha,
            #     (w, h),
            #     # centerPrincipalPoint=central_pp,
            # )

            K_pix_und = self.get_K_pix()
            if central_pp:
                K_pix_und[..., 0, 2] = w / 2
                K_pix_und[..., 1, 2] = h / 2

            if same_fx_fy:
                lens = (K_pix_und[..., 0, 0] + K_pix_und[..., 1, 1]) / 2
                K_pix_und[..., 0, 0] = lens
                K_pix_und[..., 1, 1] = lens

            K_und = K_pix_und * self.unit_pixel_ratio().unsqueeze(-1)
            K_und[..., 2, 2] = 1
        else:
            K_und = self.get_K()
            K_pix_und = self.get_K_pix()

        return K_und, K_pix_und, roi_und

    def compute_undistortion_map(self):
        try:
            D = None
            if self.D_params is not None:
                D = self.D_params.numpy()

            # undistortion map
            self.undist_map = cv2.initUndistortRectifyMap(
                self.K_pix.numpy(),
                D,
                None,
                self.K_pix_und.numpy(),
                (int(self.resolution[1]), int(self.resolution[0])),
                cv2.CV_32FC1,
            )
        except:
            {}

    def cx(self):
        return self.K_params[..., 0]

    def cy(self):
        return self.K_params[..., 1]

    def fx(self):
        return self.K_params[..., 2]

    def fy(self):
        return self.K_params[..., 3]

    def resize_pixels(self, fact=None):
        self.resolution = (
            (self.resolution * fact).type(torch.LongTensor).to(self.device)
        )
        self.K_pix = self.get_K_pix().to(self.device)
        _, K_pix_und, self.roi_und = self.get_K_und()
        self.K_pix_und = K_pix_und.to(self.device)

    def pixel_unit_ratio(self):
        return self.resolution[..., 0:1] / self.sensor_size[..., 0:1]

    def unit_pixel_ratio(self):
        return self.sensor_size[..., 0:1] / self.resolution[..., 0:1]

    def lens(self):
        return torch.cat((self.fx().unsqueeze(-1), self.fy().unsqueeze(-1)), dim=-1)

    def lens_squeezed(self):
        return (self.fx() + self.fy()) / 2

    def to(self, device):
        self.device = device
        self.resolution = self.resolution.to(device)
        self.sensor_size = self.sensor_size.to(device)
        self.K_params = self.K_params.to(device)
        if self.D_params is not None:
            self.D_params = self.D_params.to(device)
        self.update_intrinsics()
        return self

    def type(self, dtype):
        self.K_params = self.K_params.to(dtype)
        if self.D_params is not None:
            self.D_params = self.D_params.to(dtype)
        self.K_und = self.K_und.to(dtype)
        self.K_pix = self.K_pix.to(dtype)
        self.K_pix_und = self.K_pix_und.to(dtype)
        self.sensor_size = self.sensor_size.to(dtype)
        self.dtype = dtype
        return self

    def uniform_scale(self, s: float, units: str = "scaled"):
        self.K_params *= s
        self.units = units
        self.sensor_size *= s
        if self.K_und is not None:
            self.K_und[..., 0, 0] *= s
            self.K_und[..., 1, 1] *= s
            self.K_und[..., :2, -1] *= s

    def undistort_image(self, img: Image):
        undistorted = cv2.remap(
            img.numpy(), self.undist_map[0], self.undist_map[1], cv2.INTER_LINEAR
        )

        img_und = Image(torch.from_numpy(undistorted))

        # x,y,w,h = self.roi_und
        # undistorted = undistorted[y:y+h, x:x+w]
        return img_und

    def change_due_to_crop(
        self, new_resolution: Tuple[int, int], crop_offset: Tuple[int, int]
    ):
        """
        Change intrinsics due to crop
        Args:
            new_resolution (Tuple[int, int]): New resolution of the image (width, height)
            crop_offset (Tuple[int, int]): Offset of the crop (x_offset, y_offset)
        """

        ratio_x = new_resolution[0] / self.resolution[1]
        ratio_y = new_resolution[1] / self.resolution[0]

        self.resolution = torch.tensor(
            (new_resolution[1], new_resolution[0]), dtype=torch.long, device=self.device
        )
        self.K_params[..., 2] -= crop_offset[0]
        self.K_params[..., 3] -= crop_offset[1]

        self.sensor_size[0] *= ratio_y
        self.sensor_size[1] *= ratio_x

        self.update_intrinsics()
        self.compute_undistortion_map()


class Camera_cv:
    def __init__(
        self,
        intrinsics=Intrinsics(),
        pose=Pose(),
        image_paths=None,
        frame=None,
        name="Unk Cam",
        load_images=None,
        device="cpu",
        dtype=torch.float32,
        resolution_drop=1.0,
    ):
        self.images_loaded = load_images
        self.device = device
        self.resolution_drop = resolution_drop
        self.name = name
        self.frame = frame
        self.pose = pose.to(device).dtype(dtype)
        self.intr = intrinsics.to(device).type(dtype)
        self.images = {}
        self.image_paths = image_paths
        self.dtype = dtype
        self.type(self.dtype)
        if self.intr.units != self.pose.units:
            raise ValueError(
                "frame units ("
                + self.pose.units
                + ") and intrinsics units ("
                + self.intr.units
                + ") must be the same"
            )
        if load_images is not None:
            self.load_images(load_images, device)

    def set_resolution_drop(self, resolution_drop):
        if self.resolution_drop != resolution_drop:
            self.resolution_drop = resolution_drop
            self.load_images(self.images_loaded)

    def clone(self, same_intr=False, same_pose=False, image_paths=None, name=None):
        if same_intr:
            new_intr = self.intr
        else:
            new_intr = cp.deepcopy(self.intr)

        if same_pose:
            new_pose = self.pose
        else:
            new_pose = cp.deepcopy(self.pose)

        if image_paths is None:
            image_paths = self.image_paths
        if name is None:
            name = self.name + "_copy"

        new_cam = Camera_cv(
            new_intr,
            new_pose,
            image_paths,
            self.frame,
            name,
            device=self.device,
            dtype=self.dtype,
        )
        return new_cam

    def type(self, dtype):
        self.pose = self.pose.dtype(dtype)
        self.intr = self.intr.type(dtype)
        self.dtype = dtype
        return self

    def to(self, device):
        self.pose = self.pose.to(device)
        self.intr = self.intr.to(device)
        self.device = device
        return self

    def get_camera_opencv(self, device=None):
        if device is None:
            device = self.device
        K = self.intr.K_pix_und.clone()
        R = self.pose.get_R_inv()
        t = self.pose.get_t_inv()
        cam_cv = Camera_opencv(K, R, t, device, self.dtype)

        return cam_cv

    def assert_image_shape(self, image):
        r_imag = list(image.shape[:2])
        r_intr = self.intr.resolution[[1, 0]].tolist()
        assert r_imag == r_intr

    def load_images(self, images=None):
        if images is None:
            images = self.image_paths.keys()

        for image_name in images:
            image_path = self.image_paths[image_name]
            if not os.path.exists(image_path):
                raise ValueError(f"{image_path} is not a valid path")
            image = Image(
                path=image_path,
                device=self.device,
                resolution_drop=self.resolution_drop,
            )
            # self.assert_image_shape(image)
            self.images[image_name] = image

    def free_images(self):
        del self.images
        self.images = {}

    def show_image(self, img_name=None, wk=0):
        image = self.get_image(img_name)
        image.show(img_name, wk)

    def get_image(self, img_name=None):
        if img_name is None:
            assert len(self.images) == 1
            return list(self.images.values())[0]

        image = self.images[img_name]
        # if torch.is_tensor(image):
        #     image = image.numpy()
        return image

    def show_images(self, wk=0):
        for name in self.images.keys():
            self.show_image(name, wk)

    def get_pixel_grid(self, n=None, device="cpu"):
        if n is None:
            n = self.intr.resolution
        offs = (self.intr.resolution / n) * 0.5
        x_range = torch.linspace(offs[0], self.intr.resolution[0] - offs[0], n[0]).to(
            device
        )
        y_range = torch.linspace(offs[1], self.intr.resolution[1] - offs[1], n[1]).to(
            device
        )
        X, Y = torch.meshgrid(x_range, y_range, indexing="ij")
        grid_pix = torch.cat((X.unsqueeze(-1), Y.unsqueeze(-1)), dim=-1)
        grid_pix = torch.trunc(grid_pix).to(torch.int32)
        return grid_pix

    def sample_rand_pixs(self, num_pixels, device="cpu"):
        pixels_idxs = torch.reshape(
            self.get_pixel_grid(device=device), (-1, len(self.intr.resolution))
        )
        perm = torch.randperm(pixels_idxs.shape[0])
        n = min(pixels_idxs.shape[0], num_pixels)
        sampl_image_idxs = pixels_idxs[perm][:n]
        return sampl_image_idxs

    # def sample_rand_pixs_in_mask( self, mask, percentage=1):
    #     self.assert_image_shape(mask)
    #     m = mask.transpose(0,1)
    #     grid = self.get_pixel_grid( device=mask.device)
    #     pixels_idxs = grid[ m>0 ]
    #     if not pixels_idxs.numel():
    #         # return torch.empty((0, 2), dtype=torch.float32, device=self.device)
    #         return None
    #     pixels_idxs = torch.reshape(pixels_idxs, (-1,len(self.intr.resolution)))
    #     perm = torch.randperm(pixels_idxs.shape[0])
    #     sampl_image_idxs = pixels_idxs[perm][:(int(len(perm)*percentage))]
    #     return sampl_image_idxs

    def sample_rand_pixs_in_mask(self, mask, n_pixs=None):
        with torch.no_grad():
            self.assert_image_shape(mask)
            m = mask.transpose(0, 1)
            grid = self.get_pixel_grid(device=mask.device)
            pixels_idxs = grid[m > 0]
            if not pixels_idxs.numel():
                # return torch.empty((0, 2), dtype=torch.float32, device=self.device)
                return None
            pixels_idxs = torch.reshape(pixels_idxs, (-1, len(self.intr.resolution)))

            if n_pixs is None:
                n_pixs = pixels_idxs.shape[0]
                sampl_image_idxs = pixels_idxs
            else:
                perm = torch.randperm(pixels_idxs.shape[0])
                n_pixs = min(n_pixs, len(perm))
                sampl_image_idxs = pixels_idxs[perm][:n_pixs]

            return sampl_image_idxs

    def get_all_rays(self):
        grid = self.get_pixel_grid()
        origin, dir = self.pix2ray(grid.view(-1, 2))
        return origin, dir

    def pix2dir(self, pix):
        # pix = pix.to(self.device)*self.intr.unit_pixel_ratio() # pixels to units
        pix = pix.clone().type(torch.float32)
        # pix = pix.clone()
        K_inv = torch.inverse(self.intr.K_pix_und).type(pix.dtype).to(pix.device)

        original_shape = list(pix.shape)
        pix_flatten = pix.view(-1, 2)

        shp = list(pix_flatten.shape)
        shp[-1] = 1
        pixels_homogeneous = torch.cat(
            (pix_flatten, torch.ones(shp, device=pix.device)), dim=-1
        )

        normalized_coordinates = torch.matmul(K_inv, pixels_homogeneous.t()).t()
        dir_norm = normalized_coordinates / torch.norm(
            normalized_coordinates, dim=-1, keepdim=True
        )
        dir_flatten = torch.matmul(dir_norm, self.pose.rotation().T.to(pix.device))

        d = dir_flatten.reshape(*(original_shape[0:-1] + [3]))

        return d

    def pix2ray(self, pix):
        dir = self.pix2dir(pix)
        origin = self.pose.location()
        origin = repeat_tensor_to_match_shape(origin, dir.shape)
        return origin, dir

    def collect_pixs_from_img(self, image, pix):
        assert pix.dtype == torch.int32
        return image[pix[:, 0], pix[:, 1], ...]

    # def get_overlayed_image( self, obj, image_name='rgb' ):
    #     # image = self.get_image(image_name)
    #     # bytes to float
    #     image = self.get_image(image_name).float()
    #     gbuffer = Renderer.diffrast(self, obj, ["mask"], with_antialiasing=True)
    #     overlayed = (gbuffer["mask"].to(image.device) + 1.0) * image
    #     # overlayed = (gbuffer["mask"].to(image.device)) * image
    #     overlayed = overlayed.clamp_(min=0.0, max=1.0).cpu()
    #     # overlayed = torch.swapaxes(overlayed, 0,1)
    #     # overlayed_img = Image(img=overlayed)
    #     # overlayed_img.img =  overlayed_img.swapped()
    #     # return overlayed_img
    #     return overlayed

    # projection
    def get_points_wrt_cam(self, points, transform_cam_pose: Optional[Pose] = None):
        assert torch.is_tensor(points)
        assert points.shape[-1] == 3
        points = points.to(self.dtype)
        if transform_cam_pose is not None:
            pose = self.pose.get_inverse_pose() * transform_cam_pose
        else:
            pose = self.pose.get_inverse_pose()
        # T = self.pose.get_T_inverse()
        R_inv = pose.rotation()
        t_inv = pose.location()

        # points_wrt_cam = torch.matmul( points, T[...,:3,:3].transpose(-2,-1) ) + T[...,:3,-1]
        points_wrt_cam = torch.matmul(
            points, R_inv.transpose(-2, -1)
        ) + t_inv.unsqueeze(-2)
        return points_wrt_cam

    def project_points_in_cam(
        self, points_wrt_cam: torch.Tensor, longtens: bool = True, und: bool = True
    ):
        assert torch.is_tensor(points_wrt_cam)
        assert points_wrt_cam.shape[-1] == 3
        K = self.intr.K_pix
        if und:
            K = self.intr.K_pix_und
        points_wrt_cam_scaled = points_wrt_cam * self.intr.pixel_unit_ratio().unsqueeze(
            -1
        )
        uv = points_wrt_cam_scaled @ torch.transpose(K, -2, -1)
        d = uv[..., 2:]
        pixels = uv[..., :2] / d
        # pixels = torch.index_select(pixels, 1, torch.LongTensor([1,0]))
        if longtens:
            pixels = torch.trunc(pixels).to(torch.int32)

        return pixels, d

    def project_points_opencv(self, points):
        # points from units to pixels
        points_wrt_cam = self.get_points_wrt_cam(points)
        points_pix = (points_wrt_cam * self.intr.pixel_unit_ratio()).numpy()
        # rvec, _ = cv2.Rodrigues(self.pose.rotation().numpy())
        rvec = np.array([[0, 0, 0]], dtype="float32")
        # tvec = (self.pose.location() * self.intr.pixel_unit_ratio()).numpy()
        tvec = np.array([[0, 0, 0]], dtype="float32")
        K = self.intr.K_pix.numpy()
        D = self.intr.D.numpy()
        # D = np.array([[0,0,0,0]], dtype='float32')
        proj_points, _ = cv2.projectPoints(points_pix, rvec, tvec, K, D)
        proj_points = np.squeeze(proj_points)
        proj_points = np.trunc(proj_points)
        proj_points = proj_points.astype("int32")
        return proj_points

    def project_points(
        self,
        points: torch.Tensor,
        longtens: bool = True,
        return_depth: bool = False,
        und: bool = True,
        transform_cam_pose: Optional[Pose] = None,
    ):
        assert torch.is_tensor(points)
        assert points.shape[-1] == 3
        points_wrt_cam = self.get_points_wrt_cam(points, transform_cam_pose)
        pixels, d = self.project_points_in_cam(
            points_wrt_cam=points_wrt_cam, longtens=longtens, und=und
        )
        if return_depth:
            return pixels, d
        else:
            return pixels

    def __distort_standard(self, points: torch.Tensor) -> torch.Tensor:
        # given D_params (5 distortion parameters), warp 2D points according to lens distortion

        k1, k2, p1, p2, k3 = torch.unbind(self.intr.D_params.unsqueeze(-2), dim=-1)
        fx, fy, cx, cy = torch.unbind(
            (self.intr.K_params * self.intr.pixel_unit_ratio()).unsqueeze(-2), dim=-1
        )

        # Normalize to camera coordinates
        x_n = (points[..., 0] - cx) / fx
        y_n = (points[..., 1] - cy) / fy

        # Compute r^2
        r2 = x_n**2 + y_n**2

        # Radial distortion factor
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

        # Compute radial and tangential distortion
        x_r = x_n * radial
        y_r = y_n * radial

        x_t = 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
        y_t = p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n

        # Apply distortions
        x_d = x_r + x_t
        y_d = y_r + y_t

        # Convert back to pixel coordinates
        u_d = x_d * fx + cx
        v_d = y_d * fy + cy

        out = torch.stack((u_d, v_d), dim=-1)

        return out

    def __distort_rational(self, points: torch.Tensor) -> torch.Tensor:
        # given D_params (14 distortion parameters), warp 2D points according to lens distortion
        k1, k2, p1, p2, k3, k4, k5, k6, _, _, _, _, _, _ = torch.unbind(
            self.intr.D_params.unsqueeze(-2), dim=-1
        )
        fx, fy, cx, cy = torch.unbind(
            (self.intr.K_params * self.intr.pixel_unit_ratio()).unsqueeze(-2), dim=-1
        )

        # Normalize points
        x_n = (points[..., 0] - cx) / fx
        y_n = (points[..., 1] - cy) / fy

        r2 = x_n**2 + y_n**2
        r4 = r2**2
        r6 = r2**3

        # Radial factor
        numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1 + k4 * r2 + k5 * r4 + k6 * r6
        radial = numerator / denominator

        # Apply radial
        x_r = x_n * radial
        y_r = y_n * radial

        # Tangential distortion
        x_t = 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n**2)
        y_t = p1 * (r2 + 2 * y_n**2) + 2 * p2 * x_n * y_n

        # Distorted normalized coords
        x_d = x_r + x_t
        y_d = y_r + y_t

        # Back to pixel coordinates
        u_d = x_d * fx + cx
        v_d = y_d * fy + cy

        return torch.stack((u_d, v_d), dim=-1)

    # def __distort_rational(self, points: torch.Tensor) -> torch.Tensor:
    #
    #     # given D_params (14 distortion parameters), warp 2D points according to lens distortion
    #     k1, k2, p1, p2, k3, k4, k5, k6, _, _, _, _, _, _ = torch.unbind(
    #         self.intr.D_params.unsqueeze(-2), dim=-1
    #     )
    #     fx, fy, cx, cy = torch.unbind(
    #         (self.intr.K_params * self.intr.pixel_unit_ratio()).unsqueeze(-2), dim=-1
    #     )
    #
    #     # Normalize points
    #     x_n = (points[..., 0] - cx) / fx
    #     y_n = (points[..., 1] - cy) / fy
    #
    #     r2 = x_n**2 + y_n**2
    #     r4 = r2**2
    #     r6 = r2**3
    #
    #     # Numerator polynomial (1 + k1*r2 + k2*r4 + k3*r6)
    #     numerator = 1 + k1 * r2 + k2 * r4 + k3 * r6
    #
    #     # Denominator polynomial (1 + k4*r2 + k5*r4 + k6*r6)
    #     denominator = 1 + k4 * r2 + k5 * r4 + k6 * r6
    #
    #     radial = numerator / denominator
    #
    #     x_d = x_n * radial
    #     y_d = y_n * radial
    #
    #     # Back to pixel coordinates
    #     u_d = x_d * fx + cx
    #     v_d = y_d * fy + cy
    #
    #     return torch.stack((u_d, v_d), dim=-1)

    def distort(self, points: torch.Tensor) -> torch.Tensor:

        assert torch.is_tensor(points)
        if self.intr.D_params is None:
            return points

        if self.intr.D_params.shape[-1] == 5:
            return self.__distort_standard(points)

        elif self.intr.D_params.shape[-1] == 14:
            return self.__distort_rational(points)

    def test_pix2ray(self):
        rows = self.intr.resolution[0]
        cols = self.intr.resolution[1]
        pixs = torch.tensor([[0, 0], [0, cols], [rows, 0], [rows, cols]])
        origin, dir = self.pix2ray(pixs)
        p = plotter
        p.plot_ray(origin, dir)
        p.plot_cam(self, 1)
        p.show()

    def project(self, points, depth_as_distance=False):
        """Project points to the view's image plane according to the equation x = K*(R*X + t).
        Args:
            points (torch.tensor): 3D Points (A x ... x Z x 3)
            depth_as_distance (bool): Whether the depths in the result are the euclidean distances to the camera center
                                      or the Z coordinates of the points in camera space.
        Returns:
            pixels (torch.tensor): Pixel coordinates of the input points in the image space and
                                   the points' depth relative to the view (A x ... x Z x 3).
        """
        #
        points_c = (
            points @ torch.transpose(self.pose.rotation(), 0, 1) + self.pose.location()
        )
        pixels = points_c @ torch.transpose(self.intr.K, 0, 1)
        pixels = pixels[..., :2] / pixels[..., 2:]
        depths = (
            points_c[..., 2:]
            if not depth_as_distance
            else torch.norm(points_c, p=2, dim=-1, keepdim=True)
        )
        return torch.cat([pixels, depths], dim=-1)

    def get_pyrender_cam(self, near, far):
        f = (self.intr.K_pix_und[0, 0] + self.intr.K_pix_und[1, 1]) / 2
        cx = self.intr.K_pix_und[0, 2]
        cy = self.intr.K_pix_und[1, 2]
        camera = pyrender.IntrinsicsCamera(
            fx=f,
            fy=f,
            cx=cx.item(),
            cy=cy.item(),
            znear=near,
            zfar=far,
        )
        return camera


# class Camera_on_sphere(Camera_cv):

#     def __init__(self, az_el, az_el_idx, K=torch.FloatTensor( [[30,0,18],[0,30,18],[0,0,1]]), pose=None, resolution=torch.LongTensor([700,700]), images=None, name="Unk Cam on sphere" ):
#         super().__init__(K=K, pose=pose, resolution=resolution, images=images, name=name)
#         self.alpha = az_el

#     def pix2eps( self, pix ):
#         assert( pix.dtype==torch.float32)
#         eps = -torch.arctan2(((pix-(self.intr.resolution/2))*self.millimeters_pixel_ratio), self.lens())
#         return eps

#     def get_sample_from_pixs( self, pixs ):
#         eps = self.pix2eps( pixs )
#         alpha = repeat_tensor_to_match_shape(self.alpha, eps.shape)
#         sample = { 'eps':eps, 'alpha':alpha }
#         return sample

#     def sample_pixels_from_err_img( self, num_pixels, show=True ):
#         pixs = sample_from_image_pdf( self.images["err"], num_pixels )

#         # show sampled pixels
#         if show:
#             show_pixs(pixs, self.images["err"].shape,wk=1)

#         pixs = torch.FloatTensor( pixs+0.5 )
#         return pixs


#     def render_ddf( self, ddf, device, wk=0, update_err=False, path_err="" , prt=False):

#         # for k,v in self.images.items():
#         #     if k=="err":
#         #         continue
#         #     show_image( k+":_gt", v.numpy(), wk )

#         with torch.no_grad():
#             grid = self.get_pixel_grid()
#             sample = self.get_sample_from_pixs( grid )
#             sample = dict_to_device(sample, device)
#             output = ddf.forward( sample ).detach().cpu()
#             count = 0
#             errs = []
#             for k,v in self.images.items():
#                 if k=="err":
#                     continue
#                 o = output[...,count:count+v.shape[-1]]
#                 count += v.shape[-1]
#                 errs.append(torch.abs(o-v))
#                 show_image( k, o.numpy(), wk )

#             if update_err:
#                 err = torch.cat( errs, dim=-1 )
#                 err = torch.norm(err, dim=-1)
#                 if prt:
#                     print("err: "+str(torch.sum(err).item()))
#                 cv2.imwrite(path_err+"/"+self.name+".png", err.numpy()*255)
#                 show_image("err", err, wk)

if __name__ == "__main__":
    # c = Camera_on_sphere()
    c = Camera_cv()
    # c.sample_rand_pixs( 10 )
    c.pose.rotate_by_euler(eul(torch.FloatTensor([math.pi / 4, math.pi / 4, 0])))
    # c.pose.rotate_by_euler(eul(torch.FloatTensor([math.pi/4,0,0])))
    c.pose.set_location(torch.FloatTensor([0.5, 0.2, -0.1]))
    # pixs = c.get_pixel_grid(  )
    # pixs = c.sample_rand_pixs( 10 )
    mask = torch.ones(tuple(c.intr.resolution[[1, 0]]))
    mask[: int(mask.shape[0] * 0.25), : int(mask.shape[1] * 0.75)] = 0
    pixs = c.sample_rand_pixs_in_mask(mask)  # test mask
    # pixs = c.sample_pixels( 10 )
    origin, dir = c.pix2ray(pixs)
    p = plotter
    p.plot_ray(origin, dir)
    p.plot_cam(c, 1)
    p.plot_frame(c.pose)
    p.show()
