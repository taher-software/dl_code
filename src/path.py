import os


def join_paths(*parts):
    string_parts = [str(part) for part in parts]
    return os.path.join(*string_parts)