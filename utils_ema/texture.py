import torch

try:
    from .image import Image
except:
    from image import Image


# extend the Image class
class Texture(Image):
    def __init__(self, res, n_channels):

        super().__init__(img=torch.zeros([res, res] + [n_channels]))
        self.res = res
        self.n_channels = n_channels

    @classmethod
    def init_from_uvs(cls, uvs, colors, res):
        n_channels = colors.shape[-1]
        texture = cls(res, n_channels)
        texture.set_texture_vals(uvs, colors)
        return texture

    @classmethod
    def init_from_img(cls, img):
        texture = cls(img.shape[0], img.shape[-1])
        texture.img = img
        return texture

    @classmethod
    def init_from_path(cls, path):
        img = Image(path=path)
        return cls.init_from_img(img.img)

    def set_texture_vals(self, uvs, colors):
        # convert uvs to pixel coordinates
        pixels = (uvs * (self.res - 1)).long()
        self.img[pixels[:, 1], pixels[:, 0], :] = colors
        self.img = self.img.flip(0)

    def save(self, path):
        super().save(path)

    def load(self, path):
        super().__init__(path=path)
