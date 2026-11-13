"""
Preprocess entry point: runs the adapter of config/dataset/<name>.yaml.
Usage:
    python preprocess.py droid                                  # all DROID episodes
    python preprocess.py agibot limit=5 multi_processing=false
    python preprocess.py droid dataset.episode_txt_list=/path/to/episodes.txt
"""

import argparse

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig


def main():
    parser = argparse.ArgumentParser(description="Unified preprocessing entry point.")
    parser.add_argument("dataset", type=str, default="droid", nargs="?",
                        help="Name of the dataset to preprocess (maps to config/dataset/<name>.yaml)")
    args, overrides = parser.parse_known_args()

    overrides.append(f"dataset={args.dataset}")

    with initialize(config_path="config", version_base=None):
        cfg: DictConfig = compose(config_name="config", overrides=overrides)

    preprocessor = instantiate({"_target_": cfg.dataset._target_}, cfg)
    preprocessor.run()


if __name__ == "__main__":
    main()
