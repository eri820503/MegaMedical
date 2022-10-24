from pylot.util import ThunderDB, autoload
from tqdm import tqdm
import einops as E
import pathlib
import numpy as np


def rearrange_maxslice_imgs(path):
    with ThunderDB.open(path, "c") as db:
        subjects = db["_subjects"]
        a = db["_attrs"]
        res = a["resolution"]
        n_labels = a["n_labels"]
        for k in tqdm(subjects, leave=False):
            img, seg = db[k]
            write = False
            if img.shape == (res, res, n_labels):
                img = E.rearrange(img, "h w l -> l h w")
                write = True
            if seg.shape == (res, res, n_labels):
                seg = E.rearrange(seg, "h w l -> l h w")
                write = True
            assert img.shape in [(res, res), (n_labels, res, res)]
            assert seg.shape == (n_labels, res, res)
            if write:
                db[k] = img, seg


def patch_ver(f):
    with ThunderDB.open(f, "c") as db:
        a = db["_attrs"]
        if a["slicing"] == "v4.0":
            db["_attrs"] = {
                **db["_attrs"],
                "version": a["slicing"],
                "slicing": a["version"],
            }


def patch_precision(path):
    with ThunderDB.open(path, "c") as db:
        subjects = db["_subjects"]
        a = db["_attrs"]
        res = a["resolution"]
        n_labels = a["n_labels"]
        for k in tqdm(subjects, leave=False):
            img, seg = db[k]
            write = False
            if img.dtype == np.float64:
                img = img.astype(np.float32)
                write = True
            if seg.dtype == np.float64:
                seg = seg.astype(np.float32)
                write = True
            assert img.shape in [(res, res), (n_labels, res, res)]
            assert seg.shape == (n_labels, res, res)
            if write:
                db[k] = img, seg


def patch_popcounts_splits(f):
    with ThunderDB.open(f, "c") as db:
        a = db["_attrs"]

        root = pathlib.Path("/storage/megamedical/v4-raw/")

        # Popcounts
        poplab_file = f"popmatrix__{a['slicing']}__{a['dataset']}__{a['group']}__{a['axis']}.pickle"
        poplab_path = root / "stats_files" / poplab_file
        db["_pop_lab_amount"] = autoload(poplab_path)

        # Splits
        db["_splits"] = {
            split: autoload(
                root / f"split_files/{a['dataset']}__{a['group']}__{split}.txt"
            )
            for split in ("train", "val", "test")
        }

        # Fixing version
        db["_attrs"] = {
            **db["_attrs"],
            "version": "v4.1",
        }


def all_paths(root=pathlib.Path("/storage/megamedical/v4.1")):
    import itertools

    folders = [
        root / "res64/maxslice",
        root / "res64/midslice",
        root / "res128/maxslice",
        root / "res128/midslice",
        root / "res256/maxslice",
        root / "res256/midslice",
    ]
    paths = [
        str(x.parent)
        for x in itertools.chain.from_iterable([f.glob("**/data.mdb") for f in folders])
        if not str(x.parent).endswith(".tmp")
    ]
    return paths


def apply_patch(patch: str):
    patches = {
        "version": patch_ver,
        "channel": rearrange_maxslice_imgs,
        "popcounts_splits": patch_popcounts_splits,
    }
    assert patch in patches
    paths = all_paths()
    N = len(paths)
    patch_fn = patches[patch]

    for i, p in enumerate(paths, start=1):
        print(f"{i:02d}/{N:02d}", p)
        patch_fn(p)


if __name__ == "__main__":
    import typer

    typer.run(apply_patch)
