import zarr, s3fs

uri = "s3://conduit-data-dev/processed/windows/" \
      "Mahnani Clay_20250430_192507_windowed.zarr"

# open the store for read-write **without** the consolidated= kwarg
root = zarr.open_group(
    uri,
    mode="a",                       # read/write
    storage_options={"anon": False}
)

# rebuild the metadata blob so it lists 'time' (and everything else)
zarr.consolidate_metadata(root.store)

print("consolidated metadata rebuilt")
