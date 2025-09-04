import argparse
from tag_dqn import run_mcts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # None for default reward
    run_mcts(args.config, args.seed, None, args.output_dir)