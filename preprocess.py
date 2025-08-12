import pandas as pd
import numpy as np
import os
import tifffile as tiff
from sklearn.preprocessing import LabelEncoder
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
    label_encoder = LabelEncoder()
    df['cell_type'] = label_encoder.fit_transform(df['cell_type'])
    label_mapping = {label: encoded for encoded, label in enumerate(label_encoder.classes_)}
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
        json.dump(label_mapping, f)
    sample_to_img = {}
    unique_samples = set(df['sample_id'])

    for pair in image_seg_pairs:
        if len(pair) == 2:
            # Case 1: input_dirs mode
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
            # Case 2: data_paths mode
            img, seg, sample_id = pair
            img_name = os.path.splitext(os.path.basename(img))[0]
            if sample_id not in unique_samples:
                raise ValueError(f"Provided sample_id {sample_id} not in quantification CSV.")
        sample_to_img[sample_id] = img_name

        image = tiff.imread(img)
        if transposing:
            image = np.transpose(image, (1, 2, 0))
        tiff.imwrite(os.path.join(root_path, 'CellTypes/data/images', f'{img_name}.tiff'), image)

        segmask = tiff.imread(seg)
        tiff.imwrite(os.path.join(root_path, 'CellTypes/cells', f'{img_name}.tiff'), segmask)
        df2 = df[df['sample_id'] == sample_id].copy()
        df2 = df2[['cell_id', 'cell_type']]
        max_ids = segmask.max()
        labels_array = np.full(max_ids, -1, dtype=int)
        for _, row in df2.iterrows():
            cell_id = int(row['cell_id'])
            if 1 <= cell_id <= max_ids: 
                labels_array[cell_id - 1] = row['cell_type']
        np.savez(os.path.join(root_path, 'CellTypes/cells2labels', f'{img_name}.npz'), data=labels_array)
        all_cell_ids = pd.DataFrame({'cell_id': range(1, max_ids + 1)})
        full_mapping = all_cell_ids.merge(df2, on='cell_id', how='left')
        full_mapping['cell_type'] = full_mapping['cell_type'].fillna(-1)
        full_mapping['is_filled'] = full_mapping['cell_type'] == -1
        full_mapping.to_csv(os.path.join(root_path, 'CellTypes/mappings', f'mapping_{img_name}.csv'), index=False)

    # Now we need to filter the DataFrame to only include processed samples
    processed_samples = list(sample_to_img.keys())
    df = df[df['sample_id'].isin(processed_samples)]
    # Now we create the config.json file
    ## We first need to create the blacklist
    if blacklist:
        blacklist_cleaned = [x.strip() for x in blacklist.split(",")]
    else:
        blacklist_cleaned = []
    
    ## We need to create kfolds from the images
    y = df['cell_type'].values
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

        pre_train_images = []
        for s in train_sample_ids:
            if s in sample_to_img:
                pre_train_images.append(sample_to_img[s])
            else:
                raise ValueError(f"Sample ID {s} not found in the dataset.")
        # Now we need to split  images away for validation. If training images are 1 image, we take the same image for validation, as we are using crops from images. 

        test_images = []
        for s in test_sample_ids:
            if s in sample_to_img:
                test_images.append(sample_to_img[s])
            else:
                raise ValueError(f"Sample ID {s} not found in the dataset.")

        train_images, val_images = split_train_val(pre_train_images, val_size)

        config = {
            "crop_input_size": crop_input_size,
            "crop_size": crop_size,
            "root_dir": root_path,
            "train_set": train_images,
            "val_set": val_images,
            "test_set": test_images,
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
        if split_test:
            # Split the test set into two parts for memory reasons
            test_images_part1, test_images_part2 = train_test_split(test_images, test_size=0.5, random_state=42)
            config['test_set'] = test_images_part1
            config['test_set_part2'] = test_images_part2
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
        help="Paths to the directories for input images and segmentation masks Should be executed from command line like this: --input_dirs /path/to/images /path/to/segmentations." \
        "This will only work if the image and segmentation files have the same names and the image name is present in the sample_id column with or without the extension.",
    )
    group.add_argument(
        "--data_paths",
        type=str,
        help='Path to a CSV file with "image_path", "mask_path" and "sample_id_col_label" columns listing all pairs of images and masks and their correspoding label in the sample_id column of the quantification table.',
    )
    parser.add_argument(
        "--root_path",
        type=str,
        help="Root path for saving processed data.",
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        help="Path to the quantification CSV file.",
    )
    parser.add_argument(
        "--transposing",
        action='store_true',
        help="Whether to transpose the images before saving.",
    )
    parser.add_argument(
        "--crop_input_size",
        type=int,
        default=60,
        help="Size to crop the input images for training.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=128,
        help="Cellsighter parameter, approx. double the input crop size.",
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="Number of folds for k-fold cross-validation. Default is 5.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for training. Default is 0.001.",
    )
    parser.add_argument(
        "--to_pad",
        action='store_true',
        help="Work on the border of images"
    )
    parser.add_argument(
        "--blacklist",
        type=str,
        default=None,
        help="List of markers to be excluded from training. Should be a comma-separated string, e.g. 'marker1,marker2'."
    )
    parser.add_argument(
        "--marker_path",
        type=str,
        help="Path to a txt file with channel names."
    )
    parser.add_argument(
        "--sample_batch",
        action='store_false',
        help="Whether to sample equally from the category in each batch during training."
    )
    parser.add_argument(
        "--aug",
        action='store_false',
        help="Whether to apply data augmentation during training. Default is True"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for data loading. Default is 2."
    )
    parser.add_argument(
        "--size_data",
        type=int,
        default=None,
        help="Optional, for each cell type sample size_data samples or less if there aren't enough cells from the cell type."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training. Default is 128."
    )
    parser.add_argument(
        "--swap_train_val",
        action='store_true',
        help="Whether to swap the training and validation sets."
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Proportion of the training set to be used as validation set. Default is 0.15."
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs for training. Default is 50."
    )
    parser.add_argument(
        "--split_test",
        action='store_true',
        help="Whether to split the test set into 2 parts and thereby produce 2 config files each carrying only one of the parts for memory reasons."
    )
    parser.add_argument(
        "--hierarchy_match",
        action='store_true',
        help="Whether to match the hierarchy of the dataset."
    )
    args = parser.parse_args()

    image_seg_pairs = []
    if args.input_dirs:
        image_dir, seg_dir = args.input_dirs
        for f in sorted(os.listdir(image_dir)):
            if f.endswith(".tif") or f.endswith(".tiff"):
                image_path = os.path.join(image_dir, f)
                seg_path = os.path.join(seg_dir, f)
                if os.path.exists(seg_path):
                    image_seg_pairs.append((image_path, seg_path))
                else:
                    print(f"Segmentation file for {f} not found in {seg_dir}")
    elif args.data_paths:
        try:
            df = pd.read_csv(args.data_paths)
            if "image_path" not in df.columns or "mask_path" not in df.columns or "sample_id_col_label" not in df.columns:
                parser.error(
                    "CSV file must contain 'image_path', 'mask_path' and 'sample_id_col_label' columns."
                )
            image_seg_pairs = list(zip(df["image_path"], df["mask_path"], df["sample_id_col_label"]))
        except Exception as e:
            parser.error(f"Error reading CSV file: {e}")
    
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

        

    

