import argparse
from tag_dqn import run_tag_dqn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    run_tag_dqn(args.config, args.seed, args.output_dir)