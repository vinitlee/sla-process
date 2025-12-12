import gc
import sys
import cupy as cp


def list_cupy_arrays(limit=20):
    arrays = []
    for obj in gc.get_objects():
        if isinstance(obj, cp.ndarray):
            arrays.append(obj)

    arrays.sort(key=lambda a: a.nbytes, reverse=True)

    print(f"Found {len(arrays)} cupy.ndarrays")
    for a in arrays[:limit]:
        print(
            f"shape={a.shape} "
            f"dtype={str(a.dtype)} "
            f"{a.nbytes / 1e6:7.1f} MB  "
            f"refcount={sys.getrefcount(a)}"
        )


def list_cupy_mem_stats():
    mempool = cp.get_default_memory_pool()
    pinned = cp.get_default_pinned_memory_pool()

    print("Used bytes (currently handed out):", mempool.used_bytes())
    print("Total bytes (cached in pool):     ", mempool.total_bytes())
    print("Pinned free blocks:", pinned.n_free_blocks())

    del mempool
    del pinned
