# Copyright the author(s) of DLK.
#
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

from io import BytesIO

import fsspec
import pyarrow.parquet as pq

open = fsspec.open


def ls(path: str):
    fs_ins, prefix, path, fs_type = _get_fs_protocol(path)
    if fs_type == "file":
        return fs_ins.ls(path)
    elif fs_type == "hdfs":
        path_infos = fs_ins.ls(path)
        paths = []
        for path_info in path_infos:
            path = prefix + path_info["name"]
            if path_info.get("type") == "directory":
                path += "/"
            paths.append(path)
        return paths
    else:
        raise NotImplementedError(f"Currently, only support hdfs and file protocol.")


def mkdir(path: str, create_parents: bool = True, exist_ok: bool = False):
    fs_ins, prefix, path, fs_type = _get_fs_protocol(path)
    if fs_ins.exists(path) and not exist_ok:
        raise FileExistsError(
            f"{path} already exists. Set exist_ok to True to create it."
        )
    else:
        fs_ins.mkdir(path, create_parents)


def rm(path: str, recursive: bool = False):
    fs_ins, prefix, path, fs_type = _get_fs_protocol(path)
    if path.endswith("/") and not recursive:
        raise PermissionError(
            f"rm: cannot remove '{path}': Is a directory. Set recursive to True to remove it."
        )
    else:
        fs_ins.rm(path, recursive=recursive)


def read_parquet_meta(path: str):
    with open(path, "rb") as f:
        return pq.read_metadata(f)


def read_byte_buffer(path: str):
    with open(path, "rb") as f:
        return BytesIO(f.read())


def _get_fs_protocol(path: str):
    fs_ins, path = fsspec.core.url_to_fs(path)
    fs_ins: fsspec.AbstractFileSystem
    prefix = ""
    fs_type = "file"
    if (
        isinstance(fs_ins.protocol, tuple) and "hdfs" in fs_ins.protocol
    ) or fs_ins.protocol == "hdfs":
        prefix = "hdfs://" + fs_ins.host
        fs_type = "hdfs"
    else:
        assert (
            isinstance(fs_ins.protocol, str) and fs_ins.protocol == "file"
        ), "Currently, only support hdfs and file protocol."
    return fs_ins, prefix, path, fs_type
