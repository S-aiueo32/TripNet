import argparse
parser = argparse.ArgumentParser(description='This Program Makes the List of Triplets.')
parser.add_argument("--data_dir", type=str, help="Specify the data directory.")
parser.add_argument("--loop", type=int, default=3 ,help="(Optional) Specify the loop count.")
args = parser.parse_args()

print(args.data_dir)
