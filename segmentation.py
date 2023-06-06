import torch
import urllib
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np

def resize_image(image_path, output_path):
    image = Image.open(image_path)
    
    # # Calculate the new dimensions
    # width, height = image.size
    # new_width = (width // 512) * 512
    # new_height = (height // 512) * 512
    
    # # Resize the image
    # resized_image = image.resize((new_width, new_height))

    resized_image = image.resize((512, 1024))
    
    # Save the resized image
    resized_image.save(output_path)


def pad_image(image_path):
    # Open the image
    image = Image.open(image_path)

    # Get the current image size
    width, height = image.size

    # Check if padding is necessary
    if width == 512 and height == 512:
        return image  # No need to pad

    # Create a new blank image with the desired size
    padded_image = Image.new('RGB', (512, 512), (0, 0, 0))

    # Calculate the position to paste the original image
    x = (512 - width) // 2
    y = (512 - height) // 2

    # Paste the original image onto the new blank image
    padded_image.paste(image, (x, y))

    return padded_image

def imgSeg(filename: str) -> Image:

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    # image size must be a multiple of 512
    path = "./test_sets/images/" + filename
    # resize_image(path, path)

    # input_image = Image.open(path)
    input_image = pad_image(path)
    input_image.save(path)

    input_image = input_image.convert("RGB")
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