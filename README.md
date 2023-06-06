# 2023-AI-Final-Project
## Enhanced Image Segmentation with Iterative Image Inpainting

### Intorduction

**Image segmentation** involves dividing an image into multiple regions or segments based on certain characteristics such as color, texture, or intensity. The purpose of segmentation is to simplify the representation of an image, making it easier to analyze and understand. It is a fundamental step in various computer vision tasks, including object detection, tracking, and recognition. 

**Image inpainting**, on the other hand, is a task of reconstructing missing regions in an image.

Both segmentation and inpainting are important tasks in the field of computer vision. However, for image segmentation, if an object is partially covered by other objects, it becomes challenging to achieve accurate segmentation. Therefore, after an image is segmented, we utilize the given mask to perform inpainting, and continue to segment it with the inpainted image. After inpainting, previously hidden objects are likely to be successfully segmented after a few iterations.
We use two models for inpainting and compare their performances. (MAT and LaMa)

### Related Works
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)
  - [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch) (Github)
  - for image segmentation
- [MAT: Mask-Aware Transformer for Large Hole Image Inpainting](https://arxiv.org/abs/2203.15270)
  - [MAT](https://github.com/fenglinglwb/MAT/tree/main) (Github)
  - for image inpainting
- [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161)
  - [LaMa](https://github.com/advimman/lama) (Github)
  - another image inpainting model
- [Auto-Lama](https://github.com/andy971022/auto-lama)
  - combines object detection and image inpainting to automate object removal
