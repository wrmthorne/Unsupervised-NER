import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare CycleNER model for training.')
    parser.add_argument('--data_dir', default='./data', help='Directory containing training json data to load into dataset.')
    args = parser.parse_args()

