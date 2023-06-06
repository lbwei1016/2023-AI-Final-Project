import segmentation
import inpainting
import os
# import numpy as np
from PIL import Image
from merge_mask import merge_mask

ITERATION = 2
SIZE = 512
BLACK = (0, 0, 0)

def pad_image(image_path):
    # Open the image
    image = Image.open(image_path)

    # Get the current image size
    width, height = image.size

    # Check if padding is necessary
    if width % 512 == 0 and height % 512 == 0:
        return image  # No need to pad

    # Create a new blank image with the desired size
    new_width = ((width - 1) // 512 + 1) * 512
    new_height = ((height - 1) // 512 + 1) * 512
    padded_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))

    # Calculate the position to paste the original image
    x = (new_width - width) // 2
    y = (new_height - height) // 2

    # Paste the original image onto the new blank image
    padded_image.paste(image, (x, y))

    padded_image.convert('RGB').save(image_path)
    return (new_width, new_height)
    # return padded_image

if __name__ == "__main__":
    image_path = "./test_sets/images"
    color_mask_path = "./test_sets/color_masks"
    images = os.listdir(image_path)
    color_masks = os.listdir(color_mask_path)

    for image in images:
        w, h = pad_image(f"{image_path}/{image}")
        base_mask = [BLACK for _ in range(w * h)]
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
        