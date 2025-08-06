import pandas as pd
import os
import json
import argparse
import glob

def process_evals(quant_path, root_path, fold_idx):

    pattern = os.path.join(root_path, f"test_results_fold_{fold_idx}*.csv")
    test_results_filepaths = glob.glob(pattern)
    if not test_results_filepaths:
        raise FileNotFoundError(f"No test result files found for fold {fold_idx} in path: {root_path}")
    
    list_of_dfs = [pd.read_csv(file) for file in test_results_filepaths]
    print(f"Found and loaded {len(test_results_filepaths)} files for fold {fold_idx}.")

    quant_table = pd.read_csv(quant_path)
    test_results = pd.concat(list_of_dfs, ignore_index=True)

    with open(os.path.join(root_path, f"label_mapping.json"), 'r') as f:
        mapping = json.load(f)
    inv_mapping = {v: k for k, v in mapping.items()}

    test_results['predicted_phenotype'] = test_results['label'].map(inv_mapping)

    merged = pd.merge(
        test_results[['image_id', 'cell_id', 'predicted_phenotype']],
        quant_table,
        how='right',
        left_on=['image_id', 'cell_id'],
        right_on=['sample_id', 'cell_id']
    )
    merged.dropna(subset=['predicted_phenotype'], inplace=True)


    merged.to_csv(os.path.join(root_path, f"predictions_fold_{fold_idx}.csv"), index=False)

def main():
    parser = argparse.ArgumentParser(description='Process eval outputs from Cellsighter.')
    parser.add_argument('--quant_path', type=str, required=True,
                        help='Path to the quantification table.')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path where the results are stored.')
    parser.add_argument('--fold_idx', type=int, required=True, default=0,
                        help='Fold index to process.')

    args = parser.parse_args()
    process_evals(
        quant_path=args.quant_path,
        root_path=args.root_path,
        fold_idx=args.fold_idx
    )

if __name__ == "__main__":
    main()


    

