import os

directory_path = r'/work/users/jp348bcyy/rhoneDataCube/Cube.zarr/from-zarr-1e0224d98164576ed1142f352a3253c6'


# Iterate over entries in the directory
with os.scandir(directory_path) as it:
    for entry in it:
        if entry.is_file():
            os.remove(entry.path)

# This approach doesn't store all filenames in memory at once
