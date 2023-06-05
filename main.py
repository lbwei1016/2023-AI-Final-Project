import segmentation
import inpainting
import os
# import numpy as np
from PIL import Image
from merge_mask import merge_mask

ITERATION = 2
SIZE = 512
BLACK = (0, 0, 0)

if __name__ == "__main__":
    image_path = "./test_sets/images"
    color_mask_path = "./test_sets/color_masks"
    images = os.listdir(image_path)
    color_masks = os.listdir(color_mask_path)

    for image in images:
        base_mask = [BLACK for _ in range(SIZE * SIZE)]
        for i in range(ITERATION):
            print(f"iter: {i}")
            # generate masks
            segmentation.imgSeg(image)
            print("seg good")
            print(f"merge: {color_mask_path}/{image}")
            base_mask = merge_mask(f"{color_mask_path}/{image}", base_mask)
            print("merge good")
            
            inpainting.inpaint()
            print("inpaint good")

            img = Image.new("RGB", (SIZE, SIZE))
            img.putdata(base_mask)
            img.save(f"{color_mask_path}/{i}_{image}")
        