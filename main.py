import segmentation
import inpainting
import os
# import numpy as np
from PIL import Image
from merge_mask import merge_mask

ITERATION = 3
SIZE = 512
BLACK = (0, 0, 0)

if __name__ == "__main__":
    image_path = "./test_sets/images"
    color_mask_path = "./test_sets/color_masks"
    images = os.listdir(image_path)
    color_masks = os.listdir(color_mask_path)

    base_mask = [BLACK for _ in range(SIZE * SIZE)]

    for _ in range(ITERATION):
        for image in images:
            # generate masks
            segmentation.imgSeg(image)
            base_mask = merge_mask(f"{color_mask_path}/{image}_mask.png", base_mask)
    inpainting.inpaint()

