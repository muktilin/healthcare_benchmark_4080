import functools
import gc


def cleanup_cuda_memory(cache_clearers=None):
    """Release Python references and clear PyTorch CUDA allocator cache."""
    for clearer in cache_clearers or []:
        try:
            clearer()
        except Exception as exc:
            print(f"[GPUCleanup] Cache clearer failed: {exc}")

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except ImportError:
        pass
    except Exception as exc:
        print(f"[GPUCleanup] CUDA cleanup skipped: {exc}")


def cleanup_after(fn, cache_clearers=None):
    """Wrap a regular callback and clean CUDA memory when it finishes."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            cleanup_cuda_memory(cache_clearers=cache_clearers)

    return wrapper


def cleanup_after_generator(fn, cache_clearers=None):
    """Wrap a streaming callback and clean CUDA memory after the stream ends."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        finally:
            cleanup_cuda_memory(cache_clearers=cache_clearers)

    return wrapper
