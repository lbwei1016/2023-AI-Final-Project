import segmentation
import inpainting
import os
import numpy as np
import PIL

ITERATION = 3
SIZE = 512

if __name__ == "__main__":
    images = os.listdir("./test_sets/images")
    color_masks = os.listdir("./test_sets/color_masks")

    base_mask = np.ones((SIZE, SIZE))

    # for _ in range(ITERATION):
    for image in images:
        # generate masks
        segmentation.imgSeg(image)
    inpainting.inpaint()

