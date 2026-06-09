import h5py

h5_path = "Visium_HD_FF_Human_Tonsil_feature_slice.h5"

with h5py.File(h5_path, "r") as f:
    print("=== Top-level keys ===")
    print(list(f.keys()))

    print("\n=== masks ===")
    for k in f["masks"].keys():
        obj = f["masks"][k]
        print(f"{k}: shape={getattr(obj, 'shape', None)}, dtype={getattr(obj, 'dtype', None)}")

    print("\n=== images ===")
    for k in f["images"].keys():
        obj = f["images"][k]
        print(f"{k}: shape={getattr(obj, 'shape', None)}, dtype={getattr(obj, 'dtype', None)}")

    print("\n=== features ===")
    for k in f["features"].keys():
        obj = f["features"][k]
        if isinstance(obj, h5py.Dataset):
            print(f"{k}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{k}: group")

    print("\n=== feature_slices ===")
    slice_ids = list(f["feature_slices"].keys())
    print("first 10 ids:", slice_ids[:10])
    print("total:", len(slice_ids))

    first_id = slice_ids[0]
    print(f"\n=== first feature_slice: {first_id} ===")
    g = f["feature_slices"][first_id]
    print("keys:", list(g.keys()))

    for k in g.keys():
        obj = g[k]
        if isinstance(obj, h5py.Dataset):
            print(f"{k}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{k}: group, first keys={list(obj.keys())[:20]}")
        else:
            print(f"{k}: unknown type")
