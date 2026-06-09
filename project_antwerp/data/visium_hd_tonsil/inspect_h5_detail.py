import h5py

h5_path = "Visium_HD_FF_Human_Tonsil_feature_slice.h5"

with h5py.File(h5_path, "r") as f:
    print("=== masks detail ===")
    for k in f["masks"].keys():
        g = f["masks"][k]
        print(f"\n[masks/{k}]")
        print("type:", type(g))
        print("keys:", list(g.keys()))
        for kk in g.keys():
            obj = g[kk]
            if isinstance(obj, h5py.Dataset):
                print(f"  {kk}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {kk}: group")

    print("\n=== images detail ===")
    for k in f["images"].keys():
        g = f["images"][k]
        print(f"\n[images/{k}]")
        print("type:", type(g))
        print("keys:", list(g.keys()))
        for kk in g.keys():
            obj = g[kk]
            if isinstance(obj, h5py.Dataset):
                print(f"  {kk}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {kk}: group")

    print("\n=== first 3 feature_slices detail ===")
    slice_ids = list(f["feature_slices"].keys())[:3]
    for sid in slice_ids:
        g = f["feature_slices"][sid]
        print(f"\n[feature_slices/{sid}]")
        for kk in g.keys():
            obj = g[kk]
            if isinstance(obj, h5py.Dataset):
                print(f"  {kk}: shape={obj.shape}, dtype={obj.dtype}, first5={obj[:5]}")
            else:
                print(f"  {kk}: group")
