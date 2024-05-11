import numpy as np
from skimage.io import imread
from skimage.io import imsave
import torch
import torch.nn
from torchvision import models, transforms
from PIL import Image
import torch.optim as optim

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img)


def deprocess(img):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Lambda(rescale),
        transforms.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("imagenet1000_clsid_to_human.txt") as f:
        idx2label = eval(f.read())

    alexnet = models.alexnet(pretrained=True).to(device)

    y_target = torch.LongTensor([859]).to(device)  # toaster
    num_steps = 50

    img = (imread('truck-224.png')[:, :, :3]).astype(np.uint8)
    image_tensor = preprocess(Image.fromarray(img))
    image_tensor = image_tensor.unsqueeze(0).to(device)
    optimizer = optim.Adam([image_tensor], lr = 0.001, betas = (0.9, 0.999))

    mseloss = torch.nn.MSELoss(reduction='sum')
    loss = torch.nn.CrossEntropyLoss()
    for i in range(num_steps):
        optimizer.zero_grad()
        #zero_gradients(img_variable)
        output = alexnet.forward(image_tensor)
        loss_cal = loss(output, y_target)
        penalty_on_input_change = mseloss(image_tensor.data, image_tensor)
        loss_cal += 0.00003 * penalty_on_input_change  # regularize
        loss_cal.backward()
        optimizer.step()
        #img_variable.data -= 0.1 * img_variable.grad.data

    img_np = deprocess(image_tensor.cpu().data)
    img_np = np.asarray(img_np).astype(np.uint8)
    imsave('truck-1.png', img_np)


if __name__ == "__main__":
    main()
