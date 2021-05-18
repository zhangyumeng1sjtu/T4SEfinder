import os
import argparse

import pandas as pd


def read_pred_results(result_dir, method):
    dfs = []
    dir = os.path.join(result_dir, method)
    for file in os.listdir(dir):
        if file.endswith(".txt"):
            df = pd.read_table(os.path.join(dir, file), header=None)
            df = df.iloc[:, :2]
            df.columns = ['name', method+'_'+os.path.basename(file).split(".")[0]]
            df = df.set_index('name')
            dfs.append(df)
    return dfs


def main():
    parser = argparse.ArgumentParser(description="Summarize the output prediction results.")

    parser.add_argument('-i', '--input_result_dir', type=str, required=True)
    parser.add_argument('-m', '--methods', nargs='+', type=str, default=['tapebert_mlp', 'tapebert_svm', 'pssm_cnn', 'hybrid_bilstm'])
    parser.add_argument('-o', '--output_file', type=str, default="summary.csv")

    args = parser.parse_args()

    final_df = []
    for method in args.methods:
        pred_dfs = read_pred_results(args.input_result_dir, method)
        final_df.extend(pred_dfs)

    result = pd.concat(final_df, axis=1)
    result.to_csv(os.path.join(args.input_result_dir, args.output_file))


if __name__ == "__main__":
    main()
