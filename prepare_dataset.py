"""Prepare and cache sampled graphs, supervision splits and modality masks."""
import argparse
from data.load_data import DATASET_TASKS, prepare_dataset_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root', default='dataset')
    parser.add_argument('--dataset', choices=sorted(DATASET_TASKS), required=True)
    parser.add_argument('--rounds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--eval_nodes', type=int, default=5000)
    parser.add_argument('--train_ratios', nargs='+', type=float, default=[0.01,0.02])
    parser.add_argument('--missing_rates', nargs='+', type=float, default=[0.2,0.4,0.6])
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    prepare_dataset_cache(args.data_root, args.dataset, args.rounds, args.eval_nodes,
                           args.train_ratios, args.missing_rates, args.seed)


if __name__ == '__main__':
    main()
