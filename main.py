import segmentation
import inpainting
import os

ITERATION = 3

if __name__ == "__main__":
    files = os.listdir("./test_sets/images")

    for _ in range(ITERATION):
        for file in files:
            # generate masks
            segmentation.imgSeg(file)
        inpainting.inpaint()