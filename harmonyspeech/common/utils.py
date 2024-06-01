import uuid
from functools import lru_cache, partial


@lru_cache(maxsize=None)
def is_cpu() -> bool:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return "cpu" in version("aphrodite-engine")
    except PackageNotFoundError:
        return False


@lru_cache(maxsize=None)
def is_neuron() -> bool:
    try:
        import transformers_neuronx
    except ImportError:
        transformers_neuronx = None
    return transformers_neuronx is not None


def get_cpu_memory() -> int:
    """Returns the total CPU memory of the node in bytes."""
    return psutil.virtual_memory().total


def random_uuid() -> str:
    return str(uuid.uuid4().hex)
