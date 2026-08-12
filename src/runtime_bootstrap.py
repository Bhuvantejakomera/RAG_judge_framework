from __future__ import annotations

"""Runtime guards for Apple Silicon native libraries.

This module is intentionally tiny and import-safe so it can be pulled in before
NumPy, FAISS, torch, or tokenizers are loaded.
"""

import os
from typing import Any


def configure_native_threading() -> None:
    """Force single-threaded native runtimes before heavy imports."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def configure_torch_threading() -> None:
    """Limit torch thread pools after torch has been imported."""
    try:
        import torch
    except Exception:
        return

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def configure_faiss_threading(faiss_module: Any) -> None:
    """Limit FAISS OpenMP threads after faiss has been imported."""
    setter = getattr(faiss_module, "omp_set_num_threads", None)
    if not callable(setter):
        return
    try:
        setter(1)
    except Exception:
        pass


configure_native_threading()
