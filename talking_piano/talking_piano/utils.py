def chunk_iterator(collection, size):
    for i in range(0, len(collection), size):
        yield collection[i : i + size]  # NOQA: E203
