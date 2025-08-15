import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
import pandas as pd
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import json

torch.multiprocessing.set_sharing_strategy('file_system')


def test_epoch(model, dataloader, device=None):
    with torch.no_grad():
        model.eval()
        predicted_labels = []
        pred_probs = []
        results = {'cell_id': [], 'image_id': [], 'true_labels': [], 'predicted_labels': [], 'pred_probs': []}

        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)
            # m = m.to(device=device)
            y_pred = model(x)

            pred_probs += y_pred.detach().cpu().numpy().tolist()
            predicted_labels += y_pred.detach().cpu().numpy().argmax(1).tolist()

            results['cell_id'].extend(batch['cell_id'].detach().cpu().numpy().tolist())
            results['image_id'].extend(batch['image_id'])
            results['true_labels'].extend(batch['label'].detach().cpu().numpy().tolist())
            results['predicted_labels'].extend(np.argmax(y_pred.detach().cpu().numpy(), axis=1).tolist())
            results['pred_probs'].extend(y_pred.detach().cpu().numpy().tolist())

            print(f"Eval {i} / {len(dataloader)}        ", end='\r')
        return np.array(predicted_labels), np.array(pred_probs), pd.DataFrame.from_dict(results)

def subsample_const_size(crops, size):
    """
    sample same number of cell from each class
    """
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
    parser = argparse.ArgumentParser(description='Validation')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset result folder for cellsighter')
    parser.add_argument('--fold_id', type=str, default='fold_0')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--test_set', type=str, default='test_set', help='test set from config file')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.root, "logs", args.fold_id), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.root, "logs", args.fold_id))

    # Load config
    config_path = os.path.join(args.root, f"config.json")
    with open(config_path) as f:
        config = json.load(f)

    num_channels = sum(1 for _ in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(num_channels + 1, class_num)
    weights_path = os.path.join(args.root, "weights", f"weights_{args.fold_id}.pth")
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device).eval()


    _, test_crops = load_crops(
        config["root_dir"],
        config["channels_path"],
        config["crop_size"],
        [],
        config[f'{args.fold_id}_{args.test_set}'],
        config["to_pad"],
        blacklist_channels=config["blacklist"],
    )
    test_crops = [c for c in test_crops if c._label >= 0]
    crop_input_size = config.get("crop_input_size", 100)
    test_dataset = CellCropsDataset(test_crops, transform=val_transform(crop_input_size), mask=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    predicted_labels, pred_probs, results_df = test_epoch(model, test_loader, device=device)
    results_df = results_df.rename(columns={"image_id": "sample_id"})

    # Load the label mapping
    labels_df = pd.read_csv(os.path.join(args.root,'CellTypes','mappings', f"labels_{args.cell_type_col}.csv"))
    label_map = dict(zip(labels_df['label'], labels_df['phenotype']))
    results_df["true_phenotypes"] = results_df["true_labels"].map(label_map)
    results_df["predicted_phenotypes"] = results_df["predicted_labels"].map(label_map)



    # Save the updated results

    results_df.to_csv(os.path.join(args.root, f"test_results_{args.fold_id}_{args.test_set}.csv"), index=False)
    print(f" Saved validation results")
