import os, sys
from logging import raiseExceptions
import time
import cv2
import torch
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as T
from skimage import feature, filters
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label, center_of_mass
from pathlib import Path
import multiprocessing as mp
from utils_ema.general import get_monitor
from utils_ema.const import dict_multi_show


m = get_monitor()


class Image:

    def __init__(
        self,
        img=None,
        path=None,
        gray=False,
        resolution_drop=1.0,
        device="cpu",
        dtype=torch.float32,
        rgb_to_gbr=False,
    ):
        assert (img is None) ^ (path is None)
        self.device = device

        if img is not None:
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            self.img = img.to(device)
        if path is not None:
            self.img = torch.from_numpy(cv2.imread(path)).to(device)
            if rgb_to_gbr:
                self.img = self.img[:, :, [2, 1, 0]]
            # self.img = self.swapped()

        if len(self.img.shape) == 2:
            self.img = self.img.unsqueeze(-1)

        if len(self.img.shape) < 1 or len(self.img.shape) > 3:
            raise ValueError(
                f" {len(self.img.shape)} has to be a shape of len  3 (Height, Width, Channels) or 2 (Height, Width)"
            )

        if not self.img.shape[-1] in [1, 2, 3, 4]:
            raise ValueError(
                f" n_channels (last image dimension) has to be 1 (gray), 2 (uv), 3 (rgb) or 4 (rgba), got {self.img.shape[-1]}"
            )

        if resolution_drop != 1.0:
            self.resize(resolution_drop=resolution_drop)

        if gray:
            # if self.is_grayscale(self.img):
            self.img = self.img[..., 0]
            # self.img = self.img[...,0:1]
            # self.img = torch.mean( self.img, dim=-1, dtype=torch.uint8)

        self.dtype = self.img.dtype
        self.set_type(dtype)

    def to_sparse(self):
        self.img = self.img.to_sparse()
        return self

    def __del__(self):
        del self.img

    @staticmethod
    def from_img(img):
        return Image(img=img, dtype=img.dtype, device=img.device)

    @staticmethod
    def from_path(path):
        return Image(path=path)

    def set_type(self, dtype):
        self.img = self.type(dtype)
        self.dtype = dtype

    def apply_gain(self, gain):
        dtype = self.dtype
        self.set_type(torch.float32)
        self.img *= gain
        self.img = self.img.clip(0, 1)
        self.set_type(dtype)
        return self

    def type(self, dtype):

        if dtype == self.dtype:
            return self.img

        if (self.dtype == torch.bool and (dtype in [torch.float32, torch.float64])) or (
            self.dtype in [torch.float32, torch.float64] and (dtype == torch.bool)
        ):
            img = self.img.type(dtype)

        elif self.dtype == torch.uint8 and (
            dtype == torch.float32 or dtype == torch.float64
        ):
            img = self.img.type(dtype)
            img = img * 0.00390625

        elif (
            self.dtype == torch.float32 or self.dtype == torch.float64
        ) and dtype == torch.uint8:
            img = self.img * 255
            img = img.type(dtype)

        else:
            raise ValueError(f"{dtype} not valid type")

        return img

    def get_max_val(self):
        if self.dtype in [torch.float32, torch.float64, torch.bool]:
            return 1.0
        elif self.dtype == torch.uint8:
            return 255

    def to(self, device):
        self.device = device
        self.img = self.img.to(device)
        return self

    def swapped(self):
        return torch.swapaxes(self.img, 0, 1)

    def float(self):
        return self.type(torch.float32)

    def uint8(self):
        return self.type(torch.uint8)

    def is_grayscale(self, image):
        if image.shape[-1] == 3:
            b1 = torch.max(torch.abs(image[..., 0] - image[..., 1])) == 0
            b2 = torch.max(torch.abs(image[..., 0] - image[..., 2])) == 0
            return b1 and b2
        return False

    def get_gray_cmap(self, cmap="nipy_spectral"):
        dtype = self.img.dtype
        c = plt.get_cmap(cmap)
        gray = self.gray()
        colormap_tensor = c(gray.cpu().view(-1).numpy())
        s = gray.squeeze().shape
        rgb_tensor = (
            torch.flip(torch.from_numpy(colormap_tensor[:, :3]), dims=[-1])
            .view(tuple(list(s) + [3]))
            .to(self.device)
        )
        rgb_tensor = rgb_tensor.type(dtype)
        return Image.from_img(rgb_tensor)

    def resize(self, resolution=None, resolution_drop=None, interp=cv2.INTER_LINEAR):
        assert (resolution is None) ^ (resolution_drop is None)
        if resolution is not None:
            self.img = torch.from_numpy(self.resized(resolution, interp=interp)).to(
                self.device
            )
        elif resolution_drop is not None:
            r = self.resolution() * resolution_drop
            self.img = torch.from_numpy(
                self.resized(r.type(torch.LongTensor), interp=interp)
            ).to(self.device)
        return self

    def resolution(self):
        return torch.LongTensor([self.img.shape[0], self.img.shape[1]])

    def resized(self, resolution, interp=cv2.INTER_LINEAR):
        resized = cv2.resize(
            self.numpy(),
            (int(resolution[1]), int(resolution[0])),
            interpolation=interp,
        )
        return resized

    def clone(self):
        return Image.from_img(self.img.detach().clone())

    def gray(self, keepdim=False):
        if len(self.img.shape) > 2:
            gray = self.float()
            gray = gray.mean(dim=-1, keepdim=keepdim)
            # print(keepdim)
            # print(gray.shape)
            # return gray.to
            return gray
        else:
            return self.img

    def one2three_channels(self):
        if len(self.img.shape) == 2:
            self.img = self.img.unsqueeze(-1)
        if self.img.shape[-1] == 1:
            self.img = self.img.repeat(1, 1, 3)

    def numpy(self):
        return self.img.detach().cpu().numpy()

    def show(self, img_name="Unk", wk=0):
        # if window exists already
        try:
            cv2.getWindowProperty(img_name, cv2.WND_PROP_VISIBLE) <= 0
        except:
            cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)  # Create a named window
            cv2.resizeWindow(img_name, int(m.width / 2), int(m.height / 2))
        cv2.imshow(img_name, cv2.cvtColor(self.numpy(), cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(wk)
        return key

    def save(self, img_path, verbose=False):
        img = self.to("cpu").type(torch.uint8).numpy()
        self.save_base(img, img_path, verbose)

    @staticmethod
    def save_base(img, img_path, verbose):
        img_path = str(img_path)
        Path(img_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if verbose:
            print("saved image in: ", img_path)

    def save_parallel(self, img_path, verbose=True):
        img = self.to("cpu").type(torch.uint8).numpy()
        mp.Process(target=self.save_base, args=(img, img_path, verbose)).start()

    def get_indices_with_val(self, val):
        indices = torch.nonzero(self.img == val)
        indices = indices.to(torch.int32)
        return indices

    def get_distance_map(self, thresh, exp=1, edge_method="sobel"):
        if edge_method == "sobel":
            filtered = self.sobel()
            mask = ~((filtered.gray() > thresh).unsqueeze(-1).numpy())
        elif edge_method == "canny":
            filtered = self.canny(sigma=thresh)
            mask = ~((filtered.gray() > 0.5).unsqueeze(-1).numpy())
        else:
            raise ValueError(f"edge_method {edge_method} not valid")
        dist = distance_transform_edt(mask)
        # dist = torch.from_numpy(dist / dist.max()).type(torch.float32)
        dist = torch.from_numpy(dist).type(torch.float32)
        dist = dist**exp
        return Image.from_img(dist)

    def patchify(self, patch_size, stride_factor=0.5, device="cpu"):

        stride = int(patch_size * stride_factor)

        channels = self.img.shape[-1]
        img = self.img.to(device).permute(2, 0, 1).unsqueeze(0)
        patches = img.unfold(3, patch_size, stride).unfold(2, patch_size, stride)
        patches = patches.contiguous().view(channels, -1, patch_size, patch_size)
        patches = patches.permute(1, 0, 3, 2)
        patches = patches.permute(0, 2, 3, 1)
        # img = self.patches_to_image(
        #     patches, stride_factor, self.img.shape[1], self.img.shape[0]
        # )
        return patches

    # @staticmethod
    # def patches_to_image(
    #     patches, stride_factor, res_x, res_y, max_interp=False, device="cpu"
    # ):
    #     with torch.no_grad():
    #         p = patches.to(device)
    #         patch_size = p.shape[-2]
    #         img = torch.zeros((res_x, res_y, p.shape[-1])).to(device)
    #         stride = int(p.shape[-3] * stride_factor)
    #
    #         if max_interp:
    #             op = lambda x, y: torch.maximum(x, y)
    #             # op = lambda x, y: torch.minimum(x, y)
    #         else:
    #             op = lambda x, y: y
    #
    #         for i, patch in enumerate(p):
    #             y = (i * stride) % (res_x - stride)
    #             x = (i * stride) // (res_x - stride) * stride
    #             try:
    #                 img[x : x + patch_size, y : y + patch_size, :] = op(
    #                     img[x : x + patch_size, y : y + patch_size, :], patch
    #                 )
    #             except:
    #                 pass
    #
    #     return Image(img.cpu())

    @staticmethod
    def patches_to_image(
        patches, stride_factor, res_x, res_y, max_interp=False, device="cpu"
    ):
        with torch.no_grad():
            # Move patches to the specified device
            patches = patches.to(device)

            # Extract dimensions
            batch, patch_size, _, channels = patches.shape[-4:]
            stride = int(patch_size * stride_factor)

            # Calculate number of patches along each axis
            num_patches_x = (res_x - patch_size) // stride + 1
            num_patches_y = (res_y - patch_size) // stride + 1

            # Compute placement indices for all patches
            idx_x = torch.arange(num_patches_x, device=device) * stride
            idx_y = torch.arange(num_patches_y, device=device) * stride
            grid_x, grid_y = torch.meshgrid(idx_x, idx_y, indexing="ij")

            # Flatten indices
            grid_x = grid_x.flatten()  # Shape: [batch]
            grid_y = grid_y.flatten()  # Shape: [batch]

            # Create empty tensors for image and overlaps
            img = torch.zeros((res_x, res_y, channels), device=device)
            count = torch.zeros((res_x, res_y, 1), device=device)

            # Compute offsets for each patch
            offsets_x = grid_x.unsqueeze(1) + torch.arange(
                patch_size, device=device
            ).unsqueeze(
                0
            )  # [batch, patch_size]
            offsets_y = grid_y.unsqueeze(1) + torch.arange(
                patch_size, device=device
            ).unsqueeze(
                0
            )  # [batch, patch_size]
            offsets_x = offsets_x.unsqueeze(2).expand(
                -1, -1, patch_size
            )  # [batch, patch_size, patch_size]
            offsets_y = offsets_y.unsqueeze(1).expand(
                -1, patch_size, -1
            )  # [batch, patch_size, patch_size]

            # Flatten offsets
            offsets_x = offsets_x.flatten()
            offsets_y = offsets_y.flatten()

            # Flatten patches
            flat_patches = patches.view(-1, channels)

            # Use scatter_add to aggregate patches
            linear_indices = offsets_x * res_y + offsets_y  # Compute flat index
            img_flat = torch.zeros((res_x * res_y, channels), device=device)
            count_flat = torch.zeros((res_x * res_y, 1), device=device)

            img_flat.scatter_add_(
                0, linear_indices.unsqueeze(-1).expand_as(flat_patches), flat_patches
            )
            count_flat.scatter_add_(
                0, linear_indices.unsqueeze(-1), torch.ones_like(flat_patches[..., :1])
            )

            # Reshape back to 2D
            img = img_flat.view(res_x, res_y, channels)
            count = count_flat.view(res_x, res_y, 1)

            # Normalize by overlap count
            img = img / count.clamp(min=1)

            # # Apply max interpolation if required
            # if max_interp:
            #     max_img = torch.zeros_like(img)
            #     max_img.scatter_add_(0, linear_indices, torch.max())

            return img

    # sobel
    def sobel_diff(self, kernel_size=3):

        n_channels = self.img.shape[-1]

        # 3x3 sobel kernel
        if kernel_size == 3:
            sobel_kernel_x = torch.tensor(
                [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                dtype=torch.float32,
                device=self.device,
            )
            sobel_kernel_y = torch.tensor(
                [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
                dtype=torch.float32,
                device=self.device,
            )
        # 5x5 sobel kernel
        elif kernel_size == 5:
            sobel_kernel_x = torch.tensor(
                [
                    [-1, -2, 0, 2, 1],
                    [-4, -8, 0, 8, 4],
                    [-6, -12, 0, 12, 6],
                    [-4, -8, 0, 8, 4],
                    [-1, -2, 0, 2, 1],
                ],
                dtype=torch.float32,
                device=self.device,
            )
            sobel_kernel_y = torch.tensor(
                [
                    [-1, -4, -6, -4, -1],
                    [-2, -8, -12, -8, -2],
                    [0, 0, 0, 0, 0],
                    [2, 8, 12, 8, 2],
                    [1, 4, 6, 4, 1],
                ],
                dtype=torch.float32,
                device=self.device,
            )

        # Reshape the kernels to match the 4D input format expected by F.conv2d
        sobel_kernel_x = sobel_kernel_x.view(
            1, 1, kernel_size, kernel_size
        )  # Shape: (out_channels, in_channels, kernel_height, kernel_width)
        sobel_kernel_y = sobel_kernel_y.view(1, 1, kernel_size, kernel_size)
        sobel_kernel_x = sobel_kernel_x.repeat(
            n_channels, 1, 1, 1
        )  # shape will be (3, 1, 3, 3)
        sobel_kernel_y = sobel_kernel_y.repeat(
            n_channels, 1, 1, 1
        )  # shape will be (3, 1, 3, 3)

        # swap first and last dimension, and add batch dim
        image = self.img.permute(2, 0, 1).unsqueeze(0)

        # Apply the Sobel kernels using 2D convolution
        grad_x = F.conv2d(
            image, sobel_kernel_x, padding=int(kernel_size / 2), groups=n_channels
        )
        grad_y = F.conv2d(
            image, sobel_kernel_y, padding=int(kernel_size / 2), groups=n_channels
        )

        # Compute the gradient magnitude (eps is added to avoid to break gradients)
        # grad_magnitude = torch.sqrt(
        #     (torch.pow(grad_x, 2) + torch.pow(grad_y, 2)) + 1.0e-8
        # )
        grad_magnitude = torch.abs(grad_x).mean(dim=1) + torch.abs(grad_y).mean(dim=1)
        img = grad_magnitude.permute(1, 2, 0)

        sobel_img = Image.from_img(img)
        # sobel_img.show(wk=1)

        return sobel_img

    def inpaint(self, mask=None):
        if mask is None:
            mask = (self.img.sum(dim=-1) == 0).unsqueeze(-1)
            # mask = self.img == 0

        # inpaint with cv2
        img = self.type(torch.uint8).numpy()
        mask = mask.type(torch.uint8).numpy()
        # img = cv2.inpaint(self.numpy(), mask.numpy(), 3, cv2.INPAINT_TELEA)
        # img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
        return Image(img=img)

    def gamma_correction(self, gamma):
        img = self.img
        img = img**gamma
        return Image(img=img)

    def fill_black_pixels(
        self, nerby_nonzero_pixels=5, reverse=False, device=None, kernel_size=3
    ):
        if device is None:
            device = self.img.device

        tensor = self.img.to(device)
        # unsqueeze tensor depending on the dimensions
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0).float()
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).float()

        # permute
        tensor = tensor.permute(0, 3, 1, 2)

        # max pool on tensor
        tensor_sum = F.avg_pool2d(
            tensor,
            kernel_size=kernel_size,
            stride=(1),
            padding=int(kernel_size / 2),
            divisor_override=1,
        )

        # Define a kernel to check surrounding pixels
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size), dtype=torch.float32, device=tensor.device
        )
        kernel = kernel.repeat(tensor.shape[1], 1, 1, 1)  # 1 filter for 3 channels

        # Perform 2D convolution to count non-zero neighbors
        neighbor_count = F.conv2d(
            (tensor > 0).float(),
            kernel,
            padding=int(kernel_size / 2),
            groups=tensor.shape[1],
        )

        filled_tensor = tensor.clone()
        if reverse:
            condition = (neighbor_count <= 9 - nerby_nonzero_pixels) & (tensor != 0)
            filled_tensor[condition] = 0
        else:
            condition = (neighbor_count >= nerby_nonzero_pixels) & (tensor == 0)
            filled_tensor[condition] = tensor_sum[condition] / neighbor_count[condition]

        # Fill the black pixels based on the condition

        # permute back
        filled_tensor = filled_tensor.permute(0, 2, 3, 1).squeeze(0)

        return Image(filled_tensor)

    def filter_custom(self, kernel: torch.Tensor) -> torch.Tensor:
        """
        Apply a custom filter to an image using PyTorch's conv2d function.

        Parameters:
        image (torch.Tensor): Input image tensor of shape (C, H, W) or (H, W).
        kernel (torch.Tensor): 2D filter kernel tensor of shape (kH, kW).

        Returns:
        torch.Tensor: Filtered image tensor.
        """
        image = self.img
        if image.ndim == 2:  # If grayscale image, add channel dimension
            image = image.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif image.ndim == 3:  # If RGB image, add batch dimension
            image = image.unsqueeze(0)  # (C, H, W) -> (1, H, W, C)
            image = image.permute(0, 3, 1, 2)  # (1, H, W, C) -> (1, C, H, W)

        # Convert kernel to match PyTorch's expected format for conv2d
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (kH, kW) -> (1, 1, kH, kW)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)  # 1 filter for 3 channels

        # Apply convolution
        # 1 filter for 3 channels
        filtered_image = F.conv2d(image, kernel, padding=1, groups=image.shape[1])

        # Remove batch and channel dimensions if necessary
        filtered_image = filtered_image.squeeze(0)
        filtered_image = filtered_image.permute(1, 2, 0)
        return filtered_image

    def prewitt(self):
        img_new = torch.from_numpy(filters.prewitt(self.gray().numpy()))
        return Image.from_img(img_new)

    def sobel(self):
        # print(self.gray().numpy().shape)
        # img_new = torch.from_numpy(feature.canny(self.gray().numpy()))
        gn = self.gray().numpy()
        s = filters.sobel(gn)
        img_new = torch.from_numpy(s)
        return Image.from_img(img_new)

    def canny(self, sigma):
        img_new = torch.from_numpy(feature.canny(self.gray().numpy(), sigma))
        return Image.from_img(img_new)

    def max_pooling(self, kernel_size=5):
        image_in = self.img.permute(2, 0, 1).unsqueeze(0)
        image_out = F.max_pool2d(
            image_in,
            kernel_size=kernel_size,
            stride=(1),
            padding=int(kernel_size / 2),
        ).squeeze(0)
        return Image(image_out.permute(1, 2, 0))
        # image_curr = image_curr.type(torch.uint8)
        # image_curr = image_curr.squeeze(-1)
        # cv2.imshow("",image_curr.numpy())
        # cv2.waitKey(0)
        # return image_curr

    def get_pix_max_intensity(self, dtype=torch.float32):
        img = torch.mean(self.img, dim=-1, dtype=dtype)
        m = img.max()
        pix_max = torch.nonzero(img == m)
        pix_max = pix_max.flip(dims=[-1])
        return pix_max, m

    def get_pix_min_intensity(self, dtype=torch.float32):
        img = torch.mean(self.img, dim=-1, dtype=dtype)
        m = img.min()
        pix_min = torch.nonzero(img == m)
        pix_min = pix_min.flip(dims=[-1])
        return pix_min, m

    def get_intensity_mean(self, dtype=torch.float32):
        m = torch.mean(self.img, dtype=dtype)
        m = m.flip(dims=[-1])
        return m

        # print(pix_max.shape)
        # self.show()
        # flat_index = img.argmax()
        # coords = torch.unravel_index(flat_index, img.shape)
        # index = img.argmax()
        # print(index)

    def eval_bilinear(self, pixels, top_left):

        top_right = top_left + torch.tensor([1, 0], device=pixels.device)
        bottom_left = top_left + torch.tensor([0, 1], device=pixels.device)
        bottom_right = top_left + torch.tensor([1, 1], device=pixels.device)

        frac_vertical = ((pixels[:, 1] - top_left[:, 1]) * 0.5).unsqueeze(-1)
        frac_horizont = ((pixels[:, 0] - top_left[:, 0]) * 0.5).unsqueeze(-1)

        img = self.float().to(pixels.device)
        top_left_rgb = img[top_left[:, 1].int(), top_left[:, 0].int()]
        top_right_rgb = img[top_right[:, 1].int(), top_right[:, 0].int()]
        bottom_left_rgb = img[bottom_left[:, 1].int(), bottom_left[:, 0].int()]
        bottom_right_rgb = img[bottom_right[:, 1].int(), bottom_right[:, 0].int()]

        top_interpolation = (
            top_left_rgb * (1 - frac_horizont) + top_right_rgb * frac_horizont
        )
        bottom_interpolation = (
            bottom_left_rgb * (1 - frac_horizont) + bottom_right_rgb * frac_horizont
        )
        interpolated_rgb = (
            top_interpolation * (1 - frac_vertical)
            + bottom_interpolation * frac_vertical
        )

        return interpolated_rgb

    def sample_pixels(self, num_samples):
        """
        Samples pixels from a one-channel image (torch tensor) based on the pixel values as probabilities.

        Args:
        image (torch.Tensor): A one-channel image tensor.
        num_samples (int): Number of pixels to sample.

        Returns:
        torch.Tensor: Indices of sampled pixels.
        """

        if len(self.img.shape) >= 2:
            new_img = self.img.mean(dim=-1)
        else:
            new_img = self.img

        new_img = torch.pow(new_img, 2)

        # Flatten the image and normalize the pixel values to get probabilities
        flat_image = new_img.flatten()
        probabilities = flat_image / flat_image.sum()

        # Sample pixel indices based on the computed probabilities
        sampled_indices = torch.multinomial(
            probabilities, num_samples, replacement=True
        )

        # Convert flat indices to 2D indices
        rows = sampled_indices // new_img.shape[1]
        cols = sampled_indices % new_img.shape[1]

        return torch.stack((cols, rows), dim=1)

    def put_text_on_image(self, text, position, font_scale=1, color=(255, 0, 0)):
        img = self.img.numpy()
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
        )
        return Image(img)

    def draw_lines(self, origin, end, color=(255, 0, 0), thickness=2):
        img = self.img.numpy()
        for i in range(len(origin)):
            o = torch.trunc(origin[i]).type(torch.int32).detach().cpu().numpy()
            e = torch.trunc(end[i]).type(torch.int32).detach().cpu().numpy()
            cv2.line(img, o, e, color, thickness)
        return Image(img)

    def draw_circles(self, centers, radius=3, color=(255, 0, 255), thickness=-1):
        if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers.astype(np.int32))

        # centers = centers.flip(dims=[-1])
        # centers = torch.flip(centers.type(torch.int32), dims=[-1])
        centers = torch.trunc(centers).type(torch.int32).detach().cpu().numpy()

        img = self.numpy()
        for center in centers:
            cv2.circle(img, center, radius, color, thickness)
            # cv2.circle(img, (center[0],center[1]), radius, color, thickness)
            # cv2.circle(img, (100,100), radius, color, thickness)
        return Image(img)

    def get_sharpen_image(self, sharpness_factor: float) -> np.ndarray:
        """
        Sharpens an image with a given sharpness factor.

        Parameters:
        - image: np.ndarray, the input image to be sharpened.
        - sharpness_factor: float, the level of sharpness (1.0 = no change, > 1.0 = sharper).

        Returns:
        - np.ndarray: The sharpened image.
        """
        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = self.numpy()

        # Apply the sharpening filter
        sharpened_image = cv2.filter2D(img, -1, kernel)

        # Blend the original and sharpened images based on sharpness_factor
        output_image = cv2.addWeighted(
            sharpened_image, sharpness_factor, img, 1 - sharpness_factor, 0
        )

        return Image(output_image)

    def show_points(
        self, coords=[], method="cv2", wk=1, name="unk", radius=3, color=(0, 0, 255)
    ):
        if method == "plt":
            plt.imshow(self.numpy().astype(np.uint8))  # Cast to uint8 for image display
            for y, x in coords:
                plt.plot(
                    x, y, "ro"
                )  # 'ro' for red circle; adjust color and marker as needed
            plt.show()
        elif method == "cv2":
            img = Image(self.numpy())
            # for coord in coords:
            #     img.draw_circles(coord, radius=3, color=(0, 0, 255), thickness=-1)
            img.draw_circles(coords, radius=radius, color=color, thickness=-1)
            key = img.show(img_name=name, wk=wk)
            return key

    @classmethod
    def save_multiple_images(cls, images, paths, verbose=False):
        for img, path in zip(images, paths):
            img.save(path, verbose)

    @classmethod
    def show_multiple_images(
        cls, images, wk=0, name="image", undistort=None, cams=None
    ):

        n = len(images)

        for i, img in enumerate(images):

            if undistort is not None:
                assert cams is not None
                cam = cams[i]
                img = cam.intr.undistort_image(img)

            img = img.numpy()

            cx = (m.width * 0.94) / (img.shape[0] * dict_multi_show[n - 1]["cx"])
            cy = (m.height * 0.94) / (img.shape[1] * dict_multi_show[n - 1]["cy"])
            # c = 1
            # resized = cv2.resize(img, (int(img.shape[0]*c), int(img.shape[1]*c)), interpolation= cv2.INTER_LINEAR)
            # resized = cv2.resize(img, (int(m.width/2), int(m.height/2)), interpolation= cv2.INTER_LINEAR)
            winname = name + "_" + str(i).zfill(3)
            try:
                r = cv2.getWindowProperty(winname, cv2.WND_PROP_VISIBLE)
                if r <= 0:
                    raise Exception
            except:
                cv2.namedWindow(winname, cv2.WINDOW_NORMAL)  # Create a named window
                cv2.resizeWindow(
                    winname, int(img.shape[0] * cx), int(img.shape[1] * cy)
                )
                # cv2.resizeWindow(winname, 100, 100)
                cv2.moveWindow(
                    winname,
                    int(((i % 2) == 1) * (m.width / 2)),
                    int((i > 1) * (m.height / 2)),
                )
            cv2.imshow(winname, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(wk)
        return key

    @staticmethod
    def merge_images(image_1, image_2, weight, device=None):
        if device is None:
            device = image_1.device

        assert weight >= 0 and weight <= 1
        w1 = weight
        w2 = 1 - weight
        new_img = image_1.float().to(device) * w1 + image_2.float().to(device) * w2
        new_image = Image.from_img(new_img)
        return new_image

    def is_mask(self):
        is_bool = self.dtype == torch.bool
        single_ch = len(self.img.shape) or self.img.shape[-1] == 1
        return is_bool and single_ch

    def process_clusters(self):
        """
        Process a binary mask to find clusters of contiguous pixels, their centroids,
        and create a multi-mask where each cluster is a separate channel.

        Parameters:
            mask: numpy array of shape (H, W) containing a binary mask where
                1 represents the pixels of interest (clusters), and 0 represents the background.

        Returns:
            num_clusters: The number of clusters found in the mask.
            centroids: A list of (x, y) tuples representing the centroid of each cluster.
            multi_mask: A numpy array of shape (H, W, num_clusters) where each channel represents a separate mask.
        """

        # Step 1: Label the clusters (connected components)
        # struct = torch.ones((3,3), dtype=self.img.dtype)
        # labeled_mask, num_clusters = label(
        #     self.img.cpu().numpy(), structure=struct.cpu().numpy()
        # )
        labeled_mask, num_clusters = label(self.img.cpu().numpy())

        # Step 2: Compute the centroid of each cluster
        centroids = center_of_mass(
            self.img, labeled_mask, np.arange(1, num_clusters + 1)
        )
        centroids = [(c[0], c[1]) for c in centroids]

        # Step 3: Create a multi-mask where each channel is a separate cluster
        height, width, ch = self.img.shape
        multi_mask = np.zeros((height, width, num_clusters), dtype=np.uint8)

        for cluster_id in range(1, num_clusters + 1):
            # Create a binary mask for the current cluster
            cluster_mask = (labeled_mask == cluster_id).astype(np.uint8)
            # Store it as a separate channel in the multi-mask
            multi_mask[:, :, cluster_id - 1] = cluster_mask.squeeze()

        return num_clusters, centroids, torch.from_numpy(multi_mask)

    @staticmethod
    def get_multimask_with_colormap(img, colormap_name="tab20"):
        """
        Visualizes a multi-mask by assigning colors from a matplotlib colormap to each mask (cluster)
        and displaying the result using cv2.

        Parameters:
            multi_mask: numpy array of shape (H, W, num_clusters), where each channel is a separate binary mask.
            colormap_name: Name of the matplotlib colormap to use for coloring the clusters.
        """
        height, width, num_clusters = img.shape

        # Step 1: Create an RGB image to store the result
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Step 2: Get a colormap from matplotlib
        colormap = plt.get_cmap(colormap_name, num_clusters)

        # Step 3: Assign a distinct color to each cluster using the colormap
        for cluster_idx in range(num_clusters):
            # Get the color from the colormap (returns values between 0 and 1, so scale to 0-255)
            color = np.array(colormap(cluster_idx)[:3]) * 255
            color = color.astype(np.uint8)

            # Get the mask for the current cluster
            cluster_mask = img[:, :, cluster_idx]

            # Apply the color to the corresponding pixels in the colored image
            colored_image[cluster_mask == 1] = color

        return Image(colored_image)
        # # Step 4: Display the image using OpenCV
        # cv2.imshow("Multi-mask with colormap", colored_image)
        # cv2.waitKey(0)  # Wait for a key press to close the window
