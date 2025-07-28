import pandas as pd
import numpy as np
import os
import tifffile as tiff
from sklearn.preprocessing import LabelEncoder
import json
import argparse



def process_dataset(image_seg_pairs, root_path, quant_path, transposing):
    df = pd.read_csv(quant_path)
    label_encoder = LabelEncoder()
    df['cell_type'] = label_encoder.fit_transform(df['cell_type'])
    label_mapping = {label: encoded for encoded, label in enumerate(label_encoder.classes_)}
    with open(os.path.join(root_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)
    
    for img, seg in image_seg_pairs:

        image = tiff.imread(img)
        img_name = os.path.basename(img)
        if transposing:
            image = np.transpose(image, (1, 2, 0))
        tiff.imwrite(os.path.join(root_path, 'CellTypes/data/images', img_name), image)

        segmask = tiff.imread(seg)
        tiff.imwrite(os.path.join(root_path, 'CellTypes/cells', img_name), segmask)
        df2 = df[df['image'] == img_name].copy()
        df2 = df2[['cell_id', 'cell_type']]
        max_ids = segmask.max()
        labels_array = np.full(max_ids, -1, dtype=float)
        for _, row in df2.iterrows():
            cell_id = int(row['cell_id'])
            if 1 <= cell_id <= max_ids: 
                labels_array[cell_id - 1] = row['cell_type']
        np.savez(os.path.join(root_path, 'CellTypes/cells2labels', f'{img_name.split(".")[0]}.npz'), labels=labels_array)
        all_cell_ids = pd.DataFrame({'cell_id': range(1, max_ids + 1)})
        full_mapping = all_cell_ids.merge(df2, on='cell_id', how='left')
        full_mapping['cell_type'] = full_mapping['cell_type'].fillna(-1)
        full_mapping['is_filled'] = full_mapping['cell_type'] == -1
        full_mapping.to_csv(os.path.join(root_path, 'CellTypes/mappings', f'mapping_{img_name.split(".")[0]}.csv'), index=False)
    
    # Now we create the config.json file
    #>...



def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for Cellsighter.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_dirs",
        nargs=2,
        metavar=("IMAGE_DIR", "SEG_DIR"),
        help="Paths to the directories for input images and segmentation masks Should be executed from command line like this: --input_dirs /path/to/images /path/to/segmentations." \
        "This will only work if the image and segmentation files have the same names.",
    )
    group.add_argument(
        "--data_paths",
        type=str,
        help='Path to a CSV file with "image_path" and "mask_path" columns listing all pairs of images and masks.',
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
        help="Size to crop the input images for training.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
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
            if "image_path" not in df.columns or "mask_path" not in df.columns:
                parser.error(
                    "CSV file must contain 'image_path' and 'mask_path' columns."
                )
            image_seg_pairs = list(zip(df["image_path"], df["mask_path"]))
        except Exception as e:
            parser.error(f"Error reading CSV file: {e}")
    
    process_dataset(
        image_seg_pairs=image_seg_pairs,
        root_path=args.root_path,
        quant_path=args.quant_path,
        transposing=args.transposing
    )

if __name__ == "__main__":
    main()

        

    

