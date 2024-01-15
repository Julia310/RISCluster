import shutil
import os

def copy_large_directory(src, dst):
    """
    Copy a large directory from 'src' to 'dst'
    :param src: Source directory
    :param dst: Destination directory
    """
    try:
        # Copy the directory
        shutil.copytree(src, dst)
    except Exception as e:
        print(f"Error occurred while copying: {e}")

# Replace these with your source and destination paths
source_path = '/work/users/jp348bcyy/rhoneCubeNeu/Cube.zarr'
destination_path = '/work/users/jp348bcyy/rhoneDataCube/Cube_chunked.zarr'

copy_large_directory(source_path, destination_path)

# Make sure the destination directory does not already exist
