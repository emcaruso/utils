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

    def set_texture_vals(self, uvs, colors):
        # convert uvs to pixel coordinates
        pixels = (uvs * (self.res - 1)).long()
        self.img[pixels[:, 1], pixels[:, 0], :] = colors
