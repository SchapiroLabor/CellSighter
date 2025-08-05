import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import json
import time
torch.multiprocessing.set_sharing_strategy('file_system')


def train_epoch(model, dataloader, optimizer, criterion, epoch, writer, device=None):
    model.train()
    for i, batch in enumerate(dataloader):
        x = batch['image']
        m = batch.get('mask', None)
        if m is not None:
            x = torch.cat([x, m], dim=1)
        x = x.to(device=device)
        y = batch['label'].to(device=device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y.long())
        if i % 1 == 0:
            print(f"train epoch {epoch} | iterate {i} / {len(dataloader)} | {loss.item()}                      ", end='\r')
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
        loss.backward()
        optimizer.step()

def valid_epoch(model, dataloader, criterion, epoch, writer, device=None):
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)
            y = batch['label'].to(device=device)
            y_pred = model(x)
            y = y.to(dtype=torch.long)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
            if i % 1 == 0:
                print(f"valid epoch {epoch} | iterate {i} / {len(dataloader)} | {loss.item()}                      ", end='\r')
            writer.add_scalar('Loss/valid', loss.item(), epoch * len(dataloader) + i)
        total_loss = total_loss / (i + 1)
    return total_loss

def subsample_const_size(crops, size):
    """
    sample same number of cell from each class
    """
    if size is None:
        return crops
    final_crops = []
    crops = np.array(crops)
    labels = np.array([c._label for c in crops])
    for lbl in np.unique(labels):
        indices = np.argwhere(labels == lbl).flatten()
        if (labels == lbl).sum() < size:
            chosen_indices = indices
        else:
            chosen_indices = np.random.choice(indices, size, replace=False)
        final_crops += crops[chosen_indices].tolist()
    return final_crops


def define_sampler(crops, hierarchy_match=None):
    """
    Sampler that sample from each cell category equally
    The hierarchy_match defines the cell category for each class.
    if None then each class will be category of it's own.
    """
    labels = np.array([c._label for c in crops])
    if hierarchy_match is not None:
        labels = np.array([hierarchy_match[str(l)] for l in labels])

    unique_labels = np.unique(labels)
    class_sample_count = {t: len(np.where(labels == t)[0]) for t in unique_labels}
    weight = {k: sum(class_sample_count.values()) / v for k, v in class_sample_count.items()}
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight.double(), len(samples_weight))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cell Sighter Model")
    parser.add_argument('--base_path', type=str, required=True, help='Path to the root directory of the dataset.')
    parser.add_argument('--fold_id', type=int, default=0, help='Fold identifier, e.g., 0 for fold_0')
    parser.add_argument('--time', action='store_true', help='Flag to time the training process.')
    parser.add_argument('--patience_threshold', type=int, default=20, help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--min_epochs', type=int, default=30, help='Minimum number of epochs to train before considering early stopping.')

    args = parser.parse_args()

    # Start timer if the flag is set
    start_time = time.time() if args.time else None

    # Construct paths dynamically
    config_path = os.path.join(args.base_path, f"config_fold_{args.fold_id}.json")
    weights_path = os.path.join(args.base_path, 'weights', f'fold_{args.fold_id}')
    os.makedirs(weights_path, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(weights_path, 'logs'))

    with open(config_path) as f:
        config = json.load(f)

    criterion = torch.nn.CrossEntropyLoss()
    train_crops, val_crops = load_crops(config["root_dir"],
                                        config["channels_path"],
                                        config["crop_size"],
                                        config["train_set"],
                                        config["val_set"],
                                        config["to_pad"],
                                        blacklist_channels=config["blacklist"])

    train_crops = np.array([c for c in train_crops if c._label >= 0])
    val_crops = np.array([c for c in val_crops if c._label >= 0])

    if "size_data" in config:
        train_crops = subsample_const_size(train_crops, config["size_data"])

    sampler = define_sampler(train_crops, config.get("hierarchy_match"))
    shift = 5
    crop_input_size = config.get("crop_input_size", 100)
    aug = config.get("aug", True)
    training_transform = train_transform(crop_input_size, shift) if aug else val_transform(crop_input_size)

    train_dataset = CellCropsDataset(train_crops, transform=training_transform, mask=True)
    val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]

    model = Model(num_channels + 1, class_num)
    model = model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        sampler=sampler if config["sample_batch"] else None,
        shuffle=not config["sample_batch"]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False
    )
    print(f"Train loader size: {len(train_loader)}, Validation loader size: {len(val_loader)}")

    min_loss = np.Inf
    patience = 0
    patience_threshold = args.patience_threshold
    min_epochs = args.min_epochs

    for i in range(config["epoch_max"]):
        train_epoch(model, train_loader, optimizer, criterion, device=device, epoch=i, writer=writer)
        total_loss = valid_epoch(model, val_loader, criterion, device=device, epoch=i, writer=writer)
        print(f"Epoch {i} done with validation loss: {total_loss}!")

        if total_loss < min_loss:
            patience = 0
            min_loss = total_loss
            print(f"Saving new best model at epoch {i} with validation loss: {total_loss}!")
            torch.save(model.state_dict(), os.path.join(weights_path, "weights.pth"))
        else:
            patience += 1
            if i > min_epochs and patience > patience_threshold:
                print(f'Loss has not decreased in the last {patience_threshold} epochs. Terminating training.')
                break

    # Stop timer and log the result if the flag was set
    if start_time is not None:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Log the elapsed time to a file
        time_log_path = os.path.join(args.base_path, "training_times.txt")
        log_line = f"Fold {args.fold_id} training_time: {elapsed_time:.2f} seconds\n"
        
        with open(time_log_path, "a") as f:
            f.write(log_line)
        
        print(f"Training for fold {args.fold_id} took {elapsed_time:.2f} seconds.")