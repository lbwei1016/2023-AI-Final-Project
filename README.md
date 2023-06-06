# 2023-AI-Final-Project
## Enhanced Image Segmentation with Iterative Image Inpainting

### Intorduction

**Image segmentation** involves dividing an image into multiple regions or segments based on certain characteristics such as color, texture, or intensity. The purpose of segmentation is to simplify the representation of an image, making it easier to analyze and understand. It is a fundamental step in various computer vision tasks, including object detection, tracking, and recognition. 

**Image inpainting**, on the other hand, is a task of reconstructing missing regions in an image.

Both segmentation and inpainting are important tasks in the field of computer vision. However, for image segmentation, if an object is partially covered by other objects, it becomes challenging to achieve accurate segmentation. Therefore, after an image is segmented, we utilize the given mask to perform inpainting, and continue to segment it with the inpainted image. After inpainting, previously hidden objects are likely to be successfully segmented after a few iterations.
We use two models for inpainting and compare their performances. (MAT and LaMa)