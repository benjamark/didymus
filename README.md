# didymus

Didymus is a Python tool that performs ray-tracing of STL files to produce surface representations on a regular, structured, Cartesian mesh.

The tool provides barycentric interpolation between an arbitrary number of structured representations to generate new, intermediate geometries.

## Features

- Ray-triangle intersection algorithm is GPU-acclerated using [Numba](https://numba.pydata.org/), enabling efficient computation on CUDA-capable NVIDIA GPUs.
- Distance transform of the structured representations is used for the barycentric interpolation, eliminating occlusion.
- Marching cubes used for reconstruction of STL of the intermediate geometries' structured representations.

## installation with Conda

From the root of the cloned directory, run

`conda env create -f environment.yml`

to create a Conda environment with all the dependencies.

## configuration options 

Set in `config.json`.

| Option                | Description                                           |
|-----------------------|-------------------------------------------------------|
| `ENABLE_CUDA`         | Toggle ray-tracing on GPU |
| `THREADS_PER_BLOCK`   | Set no. of CUDA threads per block |
| `resolution`          | Set ray-tracing resolution in largest dimension of the STL |
| `ENABLE_RAY_SAMPLING` | Toggle between a single ray per cell-center and a cluster of five rays per cell |
| `project_dir`         | Provide path for the project files to be written to |
| `corner_stls`         | Provide paths to the STL files that make up the corners of the barycentric map |
| `samples_per_dim`     | Set no. of samples per dimension of the simplex |
| `epsilon`             | Set interface thickness when extracting binary shell array from SDF |
| `smooth_iter`         | Set no. of iterations of Laplacian smoothing of reconstructed STL. |

