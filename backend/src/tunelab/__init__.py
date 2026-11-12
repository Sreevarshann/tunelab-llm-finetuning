"""TuneLab: genuine LoRA fine-tuning, evaluation and comparison on Apple Silicon (MLX)."""

import os

# Single-machine training. If an MPI implementation other than Open MPI is on the library path
# (e.g. Anaconda's MPICH), MLX's distributed init selects it and aborts on the first collective.
# Pointing MLX at a non-existent MPI library makes it fall back to its single-process group.
os.environ.setdefault("MLX_MPI_LIBNAME", "tunelab-single-process-no-mpi.dylib")
