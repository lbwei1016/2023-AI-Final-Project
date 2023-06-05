import segmentation
import inpainting
import os

if __name__ == "__main__":
    files = os.listdir("./test_sets/images")
    for file in files:
        segmentation.imgSeg(file)
    inpainting.inpaint()