
__all__ = ("to_parquet_dataset",)

import awkward as ak
from awkward._dispatch import high_level_function

def write_metadata(dir_path, fs, *metas, global_metadata=True):
    """Generate metadata file(s) from list of arrow metadata instances"""
    assert metas
    md = metas[0]
    with fs.open("/".join([dir_path, "_common_metadata"]), "wb") as fil:
        md.write_metadata_file(fil)
    if global_metadata:
        for meta in metas[1:]:
            md.append_row_groups(meta)
        with fs.open("/".join([dir_path, "_metadata"]), "wb") as fil:
            md.write_metadata_file(fil)


def to_parquet_dataset(directory, filenames=None, filename_extension=".parquet"):
    """
    Args:
        directory (str or Path): A local directory in which to write `_common_metadata`
            and `_metadata`, making the directory of Parquet files into a dataset.
        filenames (None or list of str or Path): If None, the `directory` will be
            recursively searched for files ending in `filename_extension` and
            sorted lexicographically. Otherwise, this explicit list of files is
            taken and row-groups are concatenated in its given order. If any
            filenames are relative, they are interpreted relative to `directory`.
        filename_extension (str): Filename extension (including `.`) to use to
            search for files recursively. Ignored if `filenames` is None.

    Creates a `_common_metadata` and a `_metadata` in a directory of Parquet files.

    The `_common_metadata` contains the schema that all files share. (If the files
    have different schemas, this function raises an exception.)

    The `_metadata` contains row-group metadata used to seek to specific row-groups
    within the multi-file dataset.
    """
#    # Dispatch
#     yield (array,)

    # Implementation
    import awkward._connect.pyarrow
    import os

    pyarrow_parquet = awkward._connect.pyarrow.import_pyarrow_parquet("ak.to_parquet_dataset")
    # fsspec = awkward._connect.pyarrow.import_fsspec("ak.to_parquet")

    # pyarrow = _import_pyarrow("ak.to_parquet.dataset")
    # import pyarrow.parquet
    # Use metadata file instead to list related files instead of searching through directory? (it can be expensive)
    directory = _regularize_path(directory)
    if not os.path.isdir(directory):
        raise ValueError(
            f"{directory!r} is not a local filesystem directory"
            + {__file__}
        )

# Need way to search for filenames, without glob?
    if filenames is None:
        import glob
        filenames = sorted(
            glob.glob(directory + f"/**/*{filename_extension}", recursive=True)
        )
    else:
        filenames = [_regularize_path(x) for x in filenames]
        filenames = [
            x if os.path.isabs(x) else os.path.join(directory, x) for x in filenames
        ]
    relpaths = [os.path.relpath(x, directory) for x in filenames]

    schema, metadata_collector = _common_parquet_schema(
        pyarrow_parquet, filenames, relpaths
    )
    pyarrow_parquet.write_metadata(schema, os.path.join(directory, "_common_metadata"))
    print("Do the schema match?", metadata_collector[0].schema.equals(metadata_collector[1].schema), metadata_collector[1].schema.equals(metadata_collector[2].schema))
    print("Column schema?", metadata_collector[0].schema.column(0).equals(metadata_collector[1].schema.column(0)), metadata_collector[1].schema.column(0).equals(metadata_collector[2].schema.column(0)))
    pyarrow_parquet.write_metadata(
        schema,
        os.path.join(directory, "_metadata"),
        metadata_collector=metadata_collector,
    )


def _regularize_path(path):
    import os
    if isinstance(path, getattr(os, "PathLike", ())):
        path = os.fspath(path)

    elif hasattr(path, "__fspath__"):
        path = path.__fspath__()

    elif path.__class__.__module__ == "pathlib":
        import pathlib

        if isinstance(path, pathlib.Path):
            path = str(path)

    if isinstance(path, str):
        path = os.path.expanduser(path)

    return path

def _common_parquet_schema(pq, filenames, relpaths): # checks that all file schema are the same
    assert len(filenames) != 0
    schema = None
    metadata_collector = []
    for filename, relpath in zip(filenames, relpaths):
        if schema is None:
            schema = pq.ParquetFile(filename).schema_arrow
            first_filename = filename
        elif not schema.equals(pq.ParquetFile(filename).schema_arrow):
            raise ValueError(
                "schema in {} differs from the first schema (in {})".format(
                    repr(filename), repr(first_filename)
                )
                + ak._util.exception_suffix(__file__)
            )
        metadata_collector.append(pq.read_metadata(filename))
        metadata_collector[-1].set_file_path(relpath)
    return schema, metadata_collector



@high_level_function()
def pyarrow_parquet_dataset(
        path_or_paths=None, 
        filesystem=None, 
        schema=None, 
        metadata=None, 
        split_row_groups=False, 
        validate_schema=True, 
        filters=None, 
        metadata_nthreads=None, 
        read_dictionary=None, 
        memory_map=False, 
        buffer_size=0, 
        partitioning='hive', 
        use_legacy_dataset=None, 
        pre_buffer=True, 
        coerce_int96_timestamp_unit=None, 
        thrift_string_size_limit=None, 
        thrift_container_size_limit=None
    ):
    _impl(path_or_paths, filesystem, schema, metadata, split_row_groups, validate_schema,
                                   filters, metadata_nthreads, read_dictionary, memory_map, buffer_size, partitioning,
                                   use_legacy_dataset, pre_buffer, coerce_int96_timestamp_unit, thrift_container_size_limit, thrift_string_size_limit)


def _impl(path_or_paths, filesystem, schema, metadata, split_row_groups, validate_schema,
                                   filters, metadata_nthreads, read_dictionary, memory_map, buffer_size, partitioning,
                                   use_legacy_dataset, pre_buffer, coerce_int96_timestamp_unit, thrift_container_size_limit, thrift_string_size_limit):
    import awkward._connect.pyarrow
    pyarrow_parquet = awkward._connect.pyarrow.import_pyarrow_parquet("ak.to_parquet_dataset")
    # import pyarrow.parquet

    pyarrow_parquet.ParquetDataset(path_or_paths, filesystem, schema, metadata, split_row_groups, validate_schema,
                                   filters, metadata_nthreads, read_dictionary, memory_map, buffer_size, partitioning,
                                   use_legacy_dataset, pre_buffer, coerce_int96_timestamp_unit, thrift_container_size_limit, thrift_string_size_limit)
    