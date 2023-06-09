import torch
import urllib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

def imgSeg(filename: str) -> Image:

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    # image size must be a multiple of 512
    path = "./test_sets/images/" + filename
    # resize_image(path, path)

    input_image = Image.open(path)
    # input_image = pad_image(path)
    # input_image = input_image.convert("RGB")
    # input_image.save(path)      


    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # print(colors)

    # plot the semantic segmentation predictions of 21 classes in each color

    r, w = output_predictions.shape

    output_predictions_copy = output_predictions.detach().clone()

    # For white-black mask
    for i in range(r):
        for j in range(w):
            if (output_predictions[i][j] > 0):
                output_predictions[i][j] = 1

    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r_color = Image.fromarray(output_predictions_copy.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette([255, 255, 255, 0, 0, 0]) # labeled object is black (0, 0, 0)
    r_color.putpalette(colors)

    # pixel = np.array(r)
    # print(pixel)

    r.convert('RGB').save(f"./test_sets/masks/{filename}")
    r_color.convert('RGB').save(f"./test_sets/color_masks/{filename}")
    # return r