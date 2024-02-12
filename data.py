from torchvision import datasets, transforms
import opendatasets as od

def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image

def download_dataset(dataset_name: str):
    # Assign the Kaggle data set URL into variable
    dataset = 'https://www.kaggle.com/ryanholbrook/dl-course-data'
    # Using opendatasets let's download the data sets
    od.download_kaggle_dataset()
    od.download(dataset)

def get_dataset(name, train=True, download=True, permutation=None):
    dataset_class = AVAILABLE_DATASETS[name]
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
        transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])

    return dataset_class(
        './datasets/{name}'.format(name=name), train=train,
        download=download, transform=dataset_transform,
    )


AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST
}

AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
}

DATASET_CONFIGS = {
    'mnist': {'size': 32, 'channels': 1, 'classes': 10}
}
