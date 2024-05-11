import numpy as np
from skimage.io import imread
from skimage.io import imsave
from torchvision import datasets, models, transforms, utils
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img, size=224):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN.tolist(), std=[1, 1, 1]),
        transforms.Lambda(lambda x: x[None]),
    ])
    return transform(img)


def deprocess(img, should_rescale=True):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[0]),
        transforms.Normalize(mean=(-MEAN).tolist(), std=[1, 1, 1]),
        transforms.Lambda(rescale) if should_rescale else transforms.Lambda(lambda x: x),
        transforms.ToPILImage(),
    ])
    return transform(img)


def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled


def make_fooling_image(X, target_y, model):
    X_fooling = X.clone()

    learning_rate = 1
    for i in range(5):
        # Forward.
        scores = model(X_fooling)
        # Current max index.
        _, index = scores.data.max(dim=1)
        # Break if we've fooled the model.
        # if index[0] == target_y:
        #    break
        # Score for the target class.
        target_score = scores[:, target_y]
        # Backward.
        target_score.backward()
        # Gradient for image.
        im_grad = X_fooling.grad.data
        # Update our image with normalised gradient.
        X_fooling.data += learning_rate * (im_grad / im_grad.norm())
        # Zero our image gradient.
        X_fooling.grad.data.zero_()
    return X_fooling


def main():
    device = 'cpu'  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("imagenet1000_clsid_to_human.txt") as f:
        idx2label = eval(f.read())

    alexnet = models.alexnet(pretrained=True)
    for param in alexnet.parameters():
        param.requires_grad = False

    target_y = 859

    img = (imread('truck.png')[:, :, :3]).astype(np.uint8)
    X_tensor = preprocess(Image.fromarray(img))
    X_fooling = make_fooling_image(X_tensor, target_y, alexnet)

    scores = alexnet(X_fooling)

    X_fooling_np = deprocess(X_fooling.clone())
    X_fooling_np = np.asarray(X_fooling_np).astype(np.uint8)

    imsave('truck-1.png', X_fooling_np)


if __name__ == "__main__":
    main()
