import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--data_dir', default='data', type=str, help='Data directory path')
    parser.add_argument('--total_epochs', default=500, type=int, help='Number of training epochs')
    parser.add_argument('--emb_size', default=1024, type=int, help='Embedding size')
    parser.add_argument('--num_layers', default=3, type=int, help='Number of GNN layers')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('--activation_slope', default=0.2, type=float, help='LeakyReLU slope')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay')
    parser.add_argument('--seed', default=1234, type=int, help='Random seed')
    parser.add_argument('--patience', default=100, type=int, help='patience')
    parser.add_argument('--device', default='cpu', type=str, help='Computing device (cpu or cuda:0, cuda:1, etc.)')
    return parser.parse_args()

args = parse_args()
