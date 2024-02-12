import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import opendatasets as od

CSV_LOCATION = "./datasets/apple-quality/apple_quality.csv"

# def _permutate_image_pixels(image, permutation):
#     if permutation is None:
#         return image

#     c, h, w = image.size()
#     image = image.view(-1, c)
#     image = image[permutation, :]
#     image.view(c, h, w)
#     return image

# def get_dataset(name, train=True, download=True, permutation=None):
#     dataset_class = AVAILABLE_DATASETS[name]
#     dataset_transform = transforms.Compose([
#         *AVAILABLE_TRANSFORMS[name],
#         transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
#     ])

#     return dataset_class(
#         './datasets/{name}'.format(name=name), train=train,
#         download=download, transform=dataset_transform,
#     )


def download_dataset(dataset_url: str):
    # Using opendatasets let's download the data sets
    od.download_kaggle_dataset(dataset_url=dataset_url, data_dir="./datasets")


# Function to zero-pad columns not used in the specific dataset
def prepare_dataset_with_zero_padding(df, start_row, end_row, included_columns):
    segment_df = df.iloc[start_row:end_row].copy()
    all_columns = [
        "Size",
        "Weight",
        "Sweetness",
        "Crunchiness",
        "Juiciness",
        "Ripeness",
        "Acidity",
        "Quality",
    ]
    for col in all_columns:
        if col not in included_columns:
            segment_df[col] = 0  # Zero padding for excluded columns
    return segment_df


# Split the dataframe into train and test sets
def get_train_test_datasets(df, start_row, end_row, included_columns, test_size=0.2):
    segment_df = prepare_dataset_with_zero_padding(
        df, start_row, end_row, included_columns
    )
    train_df, test_df = train_test_split(segment_df, test_size=test_size)
    # Identify feature columns (excluding 'Quality')
    feature_columns = [
        "Size",
        "Weight",
        "Sweetness",
        "Crunchiness",
        "Juiciness",
        "Ripeness",
        "Acidity",
    ]

    # Separate features and labels
    X_train = train_df[feature_columns]
    y_train = train_df["Quality"]
    X_test = test_df[feature_columns]
    y_test = test_df["Quality"]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit on training data
    scaler.fit(X_train)

    # Transform training and test data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the scaled arrays back to DataFrame for consistency
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=feature_columns, index=train_df.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=feature_columns, index=test_df.index
    )

    # Optionally, reattach the 'Quality' label column if needed for a combined DataFrame
    train_df_scaled = X_train_scaled_df.assign(Quality=y_train.values)
    test_df_scaled = X_test_scaled_df.assign(Quality=y_test.values)
    return train_df_scaled, test_df_scaled, scaler


def get_subdataset_subdataloader(
    df, start_row, end_row, columns, test_size=0.2, batch_size=64
):
    train_df, test_df, scaler = get_train_test_datasets(df, start_row, end_row, columns)

    # Create the datasets
    train_dataset = AppleDataset(train_df)
    test_dataset = AppleDataset(test_df)

    x = test_dataset[1]
    # Example DataLoader for training data
    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
    )

    # Example DataLoader for test data
    test_dataloader = DataLoader(
        test_dataset,
        pin_memory=True,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataset, test_dataset, train_dataloader, test_dataloader, scaler


# Custom Dataset Class
class AppleDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.copy()
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[
            idx, :-1
        ].values  # Exclude the last column (Quality labels)
        label = self.dataframe.iloc[idx, -1]  # The last column is the label
        row = torch.tensor(row, dtype=torch.float)
        if self.transform:
            row = self.transform(row)
        return row, label


def get_apple_datasets(download=True, batch_size=64):
    if download:
        download_dataset(
            "https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality"
        )
    df = pd.read_csv(CSV_LOCATION).replace({"good": 1.0, "bad": 0.0})
    df = df.drop("A_id", axis=1)
    train_datasets, test_datasets, train_dataloaders, test_dataloaders, scalers = (
        [],
        [],
        [],
        [],
        [],
    )

    # First segment
    columns = ["Size", "Weight", "Quality"]
    train_dataset, test_dataset, train_dataloader, test_dataloader, scaler = (
        get_subdataset_subdataloader(df, 0, 666, columns, batch_size=batch_size)
    )
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)
    scalers.append(scaler)
    # second segment
    columns = ["Size", "Weight", "Sweetness", "Quality"]
    train_dataset, test_dataset, train_dataloader, test_dataloader, scaler = (
        get_subdataset_subdataloader(df, 667, 666 * 2, columns, batch_size=batch_size)
    )
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)
    scalers.append(scaler)

    # third segment
    columns = ["Size", "Weight", "Sweetness", "Crunchiness", "Quality"]
    train_dataset, test_dataset, train_dataloader, test_dataloader, scaler = (
        get_subdataset_subdataloader(
            df, 666 * 2, 666 * 3, columns, batch_size=batch_size
        )
    )
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)
    scalers.append(scaler)

    # fourth segment
    columns = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Quality"]
    train_dataset, test_dataset, train_dataloader, test_dataloader, scaler = (
        get_subdataset_subdataloader(
            df, 666 * 3, 666 * 4, columns, batch_size=batch_size
        )
    )
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)
    scalers.append(scaler)

    # fifth segment
    columns = [
        "Size",
        "Weight",
        "Sweetness",
        "Crunchiness",
        "Juiciness",
        "Ripeness",
        "Quality",
    ]
    train_dataset, test_dataset, train_dataloader, test_dataloader, scaler = (
        get_subdataset_subdataloader(
            df, 666 * 4, 666 * 5, columns, batch_size=batch_size
        )
    )
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)
    scalers.append(scaler)

    # fifth segment
    columns = [
        "Size",
        "Weight",
        "Sweetness",
        "Crunchiness",
        "Juiciness",
        "Ripeness",
        "Acidity",
        "Quality",
    ]
    train_dataset, test_dataset, train_dataloader, test_dataloader, scaler = (
        get_subdataset_subdataloader(df, 666 * 5, 3999, columns, batch_size=batch_size)
    )
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    train_dataloaders.append(train_dataloader)
    test_dataloaders.append(test_dataloader)
    scalers.append(scaler)

    return train_datasets, test_datasets, train_dataloaders, test_dataloaders, scalers
