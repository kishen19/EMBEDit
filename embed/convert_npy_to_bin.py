import argparse
import numpy as np
import struct
from pathlib import Path


def write_point_range(filename, data):
    """
    Writes a 2D numpy array to a file in the PointRange format.

    Args:
        filename (str): The name of the file to write to.
        data (np.ndarray): A 2D numpy array of shape (num_points, dims).
                           The dtype should be float32.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input data must be a 2D numpy array.")

    if data.dtype != np.float32:
        print("Warning: data dtype is not float32. Casting to float32.")
        data = data.astype(np.float32)

    num_points, dims = data.shape

    with open(filename, "wb") as f:
        # Write number of points (unsigned int)
        f.write(struct.pack("I", num_points))
        # Write dimensions (unsigned int)
        f.write(struct.pack("I", dims))
        # Write the data
        f.write(data.tobytes())


def convert(npy_file_path):
    """
    Converts a .npy file to a .fbin file in the PointRange format.
    """
    npy_path = Path(npy_file_path)
    if not npy_path.exists() or npy_path.suffix != ".npy":
        print(f"Error: File not found or not a .npy file: {npy_file_path}")
        return

    embeddings = np.load(npy_file_path)

    bin_path = npy_path.with_suffix(".fbin")

    print(f"Converting {npy_path} to {bin_path}...")
    write_point_range(str(bin_path), embeddings)
    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .npy embedding file to .bin format."
    )
    parser.add_argument("npy_file", help="Path to the .npy file to convert.")
    args = parser.parse_args()

    convert(args.npy_file)
