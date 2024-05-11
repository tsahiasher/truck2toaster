import numpy as np
from skimage.io import imread
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import sys
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=[1, 1, 1]),
        transforms.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def main():
    img = sys.argv[1]
    with open("imagenet1000_clsid_to_human.txt") as f:
        idx2label = eval(f.read())

    print(img)
    alexnet = models.alexnet(pretrained=True)
    alexnet.eval()
    for param in alexnet.parameters():
        param.requires_grad = False

    img = (imread(img)[:, :, :3]).astype(np.uint8)
    X_tensor = preprocess(Image.fromarray(img))

    scores = alexnet(X_tensor)

    #sm = nn.Softmax()
    #probabilities = sm(scores).numpy()[0]
    probabilities = F.softmax(scores, dim=1).numpy()[0]
    #print(probabilities) #Converted to probabilities
    inds = np.argsort(probabilities)
    for i in range(5):
        print(inds[-1 - i], idx2label[inds[-1 - i]], probabilities[inds[-1 - i]])

if __name__ == "__main__":
    main()
