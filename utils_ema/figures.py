import matplotlib.pyplot as plt
import math


class figures():

    @classmethod
    def create_mosaic_figure(cls,images):
        num_images = len(images)
        num_rows = int(math.ceil(math.sqrt(num_images)))
        num_cols = num_rows
        fig, axs = plt.subplots(num_rows, num_rows, figsize=(2.5*num_cols, 2.5*num_rows))
        for i, image in enumerate(images):
            row = i // num_rows
            col = i % num_cols
            axs[row][col].imshow(image)
        for i in range(num_images, num_rows*num_cols):
            ax = axs.flatten()[i]
            ax.axis('off')
        return fig, axs

    @classmethod
    def show(cls):
        plt.show()

