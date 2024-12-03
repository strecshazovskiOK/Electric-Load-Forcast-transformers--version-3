import argparse
from .trainer import train_model

def main():
    parser = argparse.ArgumentParser(
        description='Train transformer model for energy consumption forecasting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to the CSV data file containing energy consumption data'
    )
    args = parser.parse_args()
    train_model(args.data)

if __name__ == '__main__':
    main()
