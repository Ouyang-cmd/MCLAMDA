import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--total_epochs', type=int)
    parser.add_argument('--emb_size', type=int)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--activation_slope', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str)
    return parser.parse_args()

args = parse_args()
