import zlib
import pickle
from typing import Any



def compress_data(data: Any) -> bytes:
    return zlib.compress(pickle.dumps(data))

def decompress_data(compressed_data: bytes) -> Any:
    return pickle.loads(zlib.decompress(compressed_data))
