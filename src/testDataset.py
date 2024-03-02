import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="A simple command line program.")
    parser.add_argument("--data_dir", type=str, default="/root/szhao/datasets/LoRA-pokemon", help="The directory containing the dataset.")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset = {}
    dataset["train"] = load_dataset("imagefolder", data_dir=args.data_dir, split="train")
    # dataset = load_dataset(args.data_dir)
    print(dataset["train"])


if __name__ == "__main__":
    main()
