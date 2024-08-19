from logging import raiseExceptions
import cv2
import torch
import os, sys
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
from PIL import Image, ImageFilter
import torchvision.transforms as T
from skimage import feature, filters

try:
    from .general import get_monitor
except:
    from general import get_monitor

m = get_monitor()


class Image:

    dict_multi_show = {
        1: {"cx": 1, "cy": 1, "m": [(0, 0)]},
        2: {"cx": 2, "cy": 1},
        3: {"cx": 2, "cy": 2},
        3: {"cx": 2, "cy": 2},
    }

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

        if not self.img.shape[-1] in [1, 3, 4]:
            raise ValueError(
                f" n_channels (last image dimension) has to be 1 (gray), 3 (rgb) or 4 (rgba), got {self.img.shape[-1]}"
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

        if self.dtype == torch.uint8 and (
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

    def to(self, device):
        self.device = device
        self.img = self.img.to(device)
        return self

    def swapped(self):
        return torch.swapaxes(self.img, 0, 1)

    def float(self):
        return self.type(torch.float32)

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
        return Image(rgb_tensor)

    def resize(self, resolution=None, resolution_drop=None):
        assert (resolution is None) ^ (resolution_drop is None)
        if resolution is not None:
            self.img = torch.from_numpy(self.resized(resolution)).to(self.device)
        elif resolution_drop is not None:
            r = self.resolution() * resolution_drop
            self.img = torch.from_numpy(self.resized(r.type(torch.LongTensor))).to(
                self.device
            )

    def resolution(self):
        return torch.LongTensor([self.img.shape[0], self.img.shape[1]])

    def resized(self, resolution):
        resized = cv2.resize(
            self.numpy(),
            (int(resolution[1]), int(resolution[0])),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized

    def clone(self):
        return Image(self.img.detach().clone())

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
        cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)  # Create a named window
        cv2.resizeWindow(img_name, int(m.width / 2), int(m.height / 2))
        cv2.imshow(img_name, self.numpy())
        key = cv2.waitKey(wk)
        return key

    def save(self, img_path, verbose=True):

        img = self.to("cpu").type(torch.uint8).numpy()
        cv2.imwrite(img_path, img)
        if verbose:
            print("saved image in: ", img_path)

    def get_indices_with_val(self, val):
        indices = torch.nonzero(self.img == val)
        indices = indices.to(torch.int32)
        return indices

    def sobel(self):
        # trans_lookup = T.Compose([
        #     T.Grayscale(),
        #     T.ToPILImage(),
        # ])
        # img_new = self.clone()
        # img_new = trans_lookup(torch.swapaxes(self.img,0,-1))
        # # img_new = img_new.filter(ImageFilter.FIND_EDGES)
        # img_new = feature.canny(img_new)
        # img_new = T.ToTensor()(img_new)  # Convert the image to a PyTorch tensor
        # img_new = torch.swapaxes(img_new,0,-1)

        # print(self.gray().numpy().shape)
        # img_new = torch.from_numpy(feature.canny(self.gray().numpy()))
        img_new = torch.from_numpy(filters.sobel(self.gray().numpy()))
        return Image(img_new)

    def max_pooling(self, kernel_size=5):
        image_curr = F.max_pool2d(
            self.img.type(torch.float32).unsqueeze(-1),
            kernel_size=kernel_size,
            stride=(1),
            padding=int(kernel_size / 2),
        ).type(torch.uint8)
        image_curr = image_curr.squeeze(-1)
        # cv2.imshow("",image_curr.numpy())
        # cv2.waitKey(0)
        return image_curr

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

        # import ipdb; ipdb.set_trace()
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

    def draw_circles(self, centers, radius=3, color=(255, 0, 255), thickness=2):
        if isinstance(centers, np.ndarray):
            centers = torch.from_numpy(centers.astype(np.int32))

        # centers = centers.flip(dims=[-1])
        centers = torch.flip(centers.type(torch.int32), dims=[-1])

        img = self.numpy()
        for center in centers:
            cv2.circle(img, center.numpy(), radius, color, thickness)
            # cv2.circle(img, (center[0],center[1]), radius, color, thickness)
            # cv2.circle(img, (100,100), radius, color, thickness)
        return Image(img)

    def show_points(self, coords=[], method="cv2", wk=1, name="unk"):
        if method == "plt":
            plt.imshow(self.numpy().astype(np.uint8))  # Cast to uint8 for image display
            for y, x in coords:
                plt.plot(
                    x, y, "ro"
                )  # 'ro' for red circle; adjust color and marker as needed
            plt.show()
        elif method == "cv2":
            img = Image(self.numpy())
            coords = torch.flip(coords, dims=[-1])
            # for coord in coords:
            #     img.draw_circles(coord, radius=3, color=(0, 0, 255), thickness=-1)
            img.draw_circles(coords, radius=3, color=(0, 0, 255), thickness=-1)
            key = img.show(img_name=name, wk=wk)
            return key

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

            cx = (m.width * 0.94) / (img.shape[0] * cls.dict_multi_show[n]["cx"])
            cy = (m.height * 0.94) / (img.shape[1] * cls.dict_multi_show[n]["cy"])
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
            cv2.imshow(winname, img)
        key = cv2.waitKey(wk)
        return key

    @staticmethod
    def merge_images(image_1, image_2, weight):
        assert weight >= 0 and weight <= 1
        w1 = weight
        w2 = 1 - weight
        new_img = image_1.float() * w1 + image_2.float() * w2
        new_image = Image(new_img)
        return new_image
