import pandas as pd
import numpy as np
import os
import tifffile as tiff
# No longer using LabelEncoder
# from sklearn.preprocessing import LabelEncoder
import json
import argparse
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

def split_train_val(train_images, val_fraction=0.15):
    """Split train_images into train and val sets, ensuring at least 1 val image."""
    n_train = len(train_images)
    if n_train == 0:
        raise ValueError("No training images available.")
    if n_train == 1:
        return train_images, train_images  # Use same image for val (crops will differ)
    elif n_train == 2:
        return [train_images[0]], [train_images[1]]  # 1 for train, 1 for val
    else:
        actual_train, val = train_test_split(train_images, test_size=val_fraction, random_state=42)
        if len(val) == 0:
            val = [actual_train.pop()]  # Move one from train to val
        return actual_train, val

def process_dataset(image_seg_pairs, root_path, quant_path, transposing, crop_input_size, crop_size, kfolds, lr, to_pad, blacklist, marker_path, sample_batch, aug, num_workers, size_data, batch_size, swap_train_val, val_size, max_epochs, split_test, hierarchy_match):
    df = pd.read_csv(quant_path)
    
    # --- Label Mapping (Deterministic, sorted approach from script 2) ---
    df['cell_type'] = df['cell_type'].astype(str)
    unique_phenos = pd.Series(df['cell_type'].unique())
    df_label_map = pd.DataFrame(unique_phenos.sort_values().reset_index(drop=True), columns=["phenotype"])
    df_label_map["label"] = df_label_map.index
    df = df.merge(df_label_map, how="left", left_on="cell_type", right_on="phenotype")
    # This column will be used in the reindex logic
    df["label_id"] = df["label"].fillna(-1).astype(int)

    label_mapping = {row['phenotype']: int(row['label']) for index, row in df_label_map.iterrows()}
    
    if hierarchy_match:
        hierarchy_match = {str(encoded): label for label, encoded in label_mapping.items()}
    else:
        hierarchy_match = None

    os.makedirs(root_path, exist_ok=True)
    os.makedirs(os.path.join(root_path, 'CellTypes/data/images'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'CellTypes/cells'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'CellTypes/cells2labels'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'CellTypes/mappings'), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'weights'), exist_ok=True)

    with open(os.path.join(root_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=4)
        
    sample_to_img = {}
    unique_samples = set(df['sample_id'])

    for pair in image_seg_pairs:
        if len(pair) == 2:
            img, seg = pair
            img_basename = os.path.basename(img)
            img_name = os.path.splitext(img_basename)[0]
            candidates = [img_name, img_basename, f'{img_name}.tif', f'{img_name}.tiff']
            matching = [cand for cand in candidates if cand in unique_samples]
            if not matching:
                raise ValueError(f"No matching sample_id for {img}.")
            if len(matching) > 1:
                raise ValueError(f"Multiple sample_id matches for {img}.")
            sample_id = matching[0]
        else:
            img, seg, sample_id = pair
            img_name = os.path.splitext(os.path.basename(img))[0]
            if sample_id not in unique_samples:
                raise ValueError(f"Provided sample_id {sample_id} not in quantification CSV.")
        
        sample_to_img[sample_id] = img_name

        image = tiff.imread(img)
        if transposing:
            image = np.transpose(image, (1, 2, 0))
        np.savez_compressed(os.path.join(root_path, 'CellTypes/data/images', f'{img_name}.npz'), data=image)

        segmask = tiff.imread(seg)
        np.savez_compressed(os.path.join(root_path, 'CellTypes/cells', f'{img_name}.npz'), data=segmask)
        
        max_cell_id_in_mask = int(segmask.max())
        
        if max_cell_id_in_mask == 0:
            print(f"Warning: No cells found in segmentation mask for {img_name}. Skipping label file creation.")
            continue
        
        ### REFACTOR: Implementing the EXACT label creation logic using set_index().reindex()
        # This block now precisely follows the methodology from the script you provided.
        
        # 1. Get the subset of quantification data for the current sample
        group = df[df['sample_id'] == sample_id].copy()
        
        # 2. Sort by cell_id (as done in the target script)
        group = group.sort_values("cell_id").reset_index(drop=True)

        # 3. The expected number of cells is determined by the mask (the primary logic from the target script)
        expected_n = max_cell_id_in_mask
        full_ids = list(range(1, expected_n + 1))

        # 4. Use set_index() and reindex() to create a complete mapping for all cells in the mask
        group = group.set_index("cell_id").reindex(full_ids, fill_value=np.nan).reset_index()
        
        # 5. Fill any cells that were in the mask but not the quant data with -1
        group["label_id"] = group["label_id"].fillna(-1).astype(int)

        # 6. Save the resulting label column to the text file
        labels_output_path = os.path.join(root_path, 'CellTypes/cells2labels', f'{img_name}.txt')
        group["label_id"].to_csv(labels_output_path, index=False, header=False)

        # For inspection, you can still save the mapping file
        group.to_csv(os.path.join(root_path, 'CellTypes/mappings', f'mapping_{img_name}.csv'), index=False)


    # --- K-Fold Splitting and Config Generation (Your original logic) ---
    processed_samples = list(sample_to_img.keys())
    df = df[df['sample_id'].isin(processed_samples)]
    
    if blacklist:
        blacklist_cleaned = [x.strip() for x in blacklist.split(",")]
    else:
        blacklist_cleaned = []
    
    y = df['label_id'].values
    groups = df['sample_id'].values
    X = np.ones((len(y), 1))
    skf = StratifiedGroupKFold(n_splits=kfolds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):
        if swap_train_val:
            train_sample_ids = np.unique(groups[test_idx])
            test_sample_ids = np.unique(groups[train_idx])
        else:
            test_sample_ids = np.unique(groups[test_idx])
            train_sample_ids = np.unique(groups[train_idx])

        pre_train_images = [sample_to_img[s] for s in train_sample_ids if s in sample_to_img]
        test_images = [sample_to_img[s] for s in test_sample_ids if s in sample_to_img]

        train_images, val_images = split_train_val(pre_train_images, val_size)

        config = {
            "crop_input_size": crop_input_size,
            "crop_size": crop_size,
            "root_dir": root_path,
            "train_set": train_images,
            "val_set": val_images,
            "num_classes": len(label_mapping),
            "epoch_max": max_epochs,
            "lr": lr,
            "blacklist": blacklist_cleaned,
            "num_workers": num_workers,
            "channels_path": marker_path,
            "weight_to_eval": "",
            "sample_batch": sample_batch,
            "to_pad": to_pad,
            "aug": aug,
            "hierarchy_match": hierarchy_match,
            "size_data": size_data,
            "batch_size": batch_size
        }
        
        if split_test and split_test > 1:
            test_image_chunks = np.array_split(test_images, split_test)
            config['test_set'] = test_image_chunks[0].tolist()

            if len(test_image_chunks) > 1:
                for i, chunk in enumerate(test_image_chunks[1:], start=2):
                    config[f'test_set_part{i}'] = chunk.tolist()
        else:
            config['test_set'] = test_images

        config_path = os.path.join(root_path, f'config_fold_{fold_idx}.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for Cellsighter.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dirs",
        nargs=2,
        metavar=("IMAGE_DIR", "SEG_DIR"),
        help="Paths to the directories for input images and segmentation masks. --input_dirs /path/to/images /path/to/segmentations. Image and segmentation files must have the same name.",
    )
    group.add_argument(
        "--data_paths",
        type=str,
        help='Path to a CSV file with "image_path", "mask_path", and "sample_id_col_label" columns.',
    )
    parser.add_argument("--root_path", type=str, required=True, help="Root path for saving processed data.")
    parser.add_argument("--quant_path", type=str, required=True, help="Path to the quantification CSV file.")
    parser.add_argument("--transposing", action='store_true', help="Transpose images from (C, H, W) to (H, W, C).")
    parser.add_argument("--crop_input_size", type=int, default=60, help="Size to crop the input images for training.")
    parser.add_argument("--crop_size", type=int, default=128, help="Cellsighter parameter, approx. double the input crop size.")
    parser.add_argument("--kfolds", type=int, default=5, help="Number of folds for k-fold cross-validation. Default is 5.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training. Default is 0.001.")
    parser.add_argument("--to_pad", action='store_true', help="Work on the border of images.")
    parser.add_argument("--blacklist", type=str, default=None, help="Comma-separated list of markers to exclude, e.g. 'marker1,marker2'.")
    parser.add_argument("--marker_path", type=str, help="Path to a txt file with channel names.")
    parser.add_argument("--sample_batch", action='store_false', help="Disable sampling equally from each category per batch.")
    parser.add_argument("--aug", action='store_true', help="Enable data augmentation during training (default). Use --no-aug to disable.", default=True)
    parser.add_argument('--no-aug', dest='aug', action='store_false', help="Disable data augmentation during training.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading. Default is 2.")
    parser.add_argument("--size_data", type=int, default=None, help="Subsample to this many cells per cell type.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training. Default is 128.")
    parser.add_argument("--swap_train_val", action='store_true', help="Swap the training and test sets within a fold (for debugging or specific validation strategies).")
    parser.add_argument("--val_size", type=float, default=0.15, help="Proportion of the training set to be used as validation set. Default is 0.15.")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum number of epochs for training. Default is 50.")
    parser.add_argument("--split_test", type=int, default=1, help="Split the test set into N parts for memory reasons. Default is 1 (no split).")
    parser.add_argument("--hierarchy_match", action='store_true', help="Create a hierarchy mapping file for the model.")
    args = parser.parse_args()

    image_seg_pairs = []
    if args.input_dirs:
        image_dir, seg_dir = args.input_dirs
        for f in sorted(os.listdir(image_dir)):
            if f.endswith((".tif", ".tiff")):
                image_path = os.path.join(image_dir, f)
                seg_path = os.path.join(seg_dir, f)
                if os.path.exists(seg_path):
                    image_seg_pairs.append((image_path, seg_path))
                else:
                    print(f"Warning: Segmentation file for {f} not found in {seg_dir}. Skipping this image.")
    elif args.data_paths:
        try:
            df_paths = pd.read_csv(args.data_paths)
            if not all(col in df_paths.columns for col in ["image_path", "mask_path", "sample_id_col_label"]):
                parser.error("CSV file for --data_paths must contain 'image_path', 'mask_path', and 'sample_id_col_label' columns.")
            image_seg_pairs = list(zip(df_paths["image_path"], df_paths["mask_path"], df_paths["sample_id_col_label"]))
        except Exception as e:
            parser.error(f"Error reading CSV file from --data_paths: {e}")
    
    process_dataset(
        image_seg_pairs=image_seg_pairs,
        root_path=args.root_path,
        quant_path=args.quant_path,
        transposing=args.transposing,
        crop_input_size=args.crop_input_size,
        crop_size=args.crop_size,
        kfolds=args.kfolds,
        lr=args.lr,
        to_pad=args.to_pad,
        blacklist=args.blacklist,
        marker_path=args.marker_path,
        sample_batch=args.sample_batch,
        aug=args.aug,
        num_workers=args.num_workers,
        size_data=args.size_data,
        batch_size=args.batch_size,
        swap_train_val=args.swap_train_val,
        val_size=args.val_size,
        max_epochs=args.max_epochs,
        split_test=args.split_test,
        hierarchy_match=args.hierarchy_match
    )

if __name__ == "__main__":
    main()