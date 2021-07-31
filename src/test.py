from typing import Tuple
import argparse
from pathlib import Path

import numpy as np


def read_npy_file(root_path: str, idx: int) -> Tuple[np.array, np.array]:
    """
    Loads a pair of .npy files (color and depth image).
    """
    img_color = Path(root_path) / "frame_color_" + str(idx) + ".npy"
    img_depth = Path(root_path) / "frame_depth_" + str(idx) + ".npy"
    return (img_color, img_depth)


def main() -> None:
    pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
		"-i",
		"--image",
		default='src/test_imgs/boja.png', 
		help = "Path to the image."
	)
    args = vars(ap.parse_args())
    main()