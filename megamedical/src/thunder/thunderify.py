from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pathlib
from typing import List
import shutil

from tqdm import tqdm
from pydantic import validate_arguments
import numpy as np
import einops as E

from pylot.util import autoload, Timer
from pylot.util import ThunderDB
from pylot.datasets.thunder import ThunderDataset
from pylot.util.future import remove_prefix, remove_suffix
from pylot.util import printc

import typer


def _load(file):
    return file, autoload(file)


def parallel_loader(files, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(_load, files)


def regroup_files(files):
    grouped = defaultdict(list)
    for f in files:
        *_, res, dataset, version, group, modality, axis, subject, file = str(f).split(
            "/"
        )
        grouped[(res, dataset, version, group, modality, axis)].append(str(f))
    return grouped


def get_label_info(root, key):
    res, dataset, version, group, modality, axis = key
    v = remove_suffix(version, "_v4.0")
    f = (
        root
        / res
        / dataset
        / "label_info"
        / group
        / f"{v}_pop_lab_amount_{axis}.pickle"
    )
    if not f.exists():
        slicing, _ = version.split("_")
        f = (
            root
            / "stats_files"
            / f"popmatrix__{slicing}__{dataset}__{group}__{axis}.pickle"
        )
    return autoload(f)


def postprocess_pair(img, seg, res, n_labels):
    if img.shape == (res, res, n_labels):
        img = E.rearrange(img, "h w l -> l h w")
    if seg.shape == (res, res, n_labels):
        seg = E.rearrange(seg, "h w l -> l h w")
    assert img.shape in [(res, res), (n_labels, res, res)]
    assert seg.shape == (n_labels, res, res), f"{seg.shape}, {(n_labels, res, res)}"
    img = img.astype(np.float32)
    seg = seg.astype(np.float32)
    return img, seg


def get_splits(root, key):
    _, dataset, version, group, modality, axis = key
    # need to hardcode res256 because splits vary per resolution
    # Hardcoded maxslice_v4.0 for the same reason
    return {
        split: autoload(root / f"split_files/{dataset}__{group}__{split}.txt")
        for split in ("train", "val", "test")
    }


def thunderify(root, key, imgs, dst, version, max_workers=16):
    with ThunderDB.open(str(dst), "c", map_size=2 ** 32) as db:
        segs = [img.replace("/img.npy", "/seg.npy") for img in imgs]
        img_loader = parallel_loader(imgs, max_workers=max_workers // 2)
        seg_loader = parallel_loader(segs, max_workers=max_workers // 2)

        splits = get_splits(root, key)
        subjects = [name.split("/")[-2] for name in imgs]
        from_splits = set(sum(splits.values(), start=[]))
        from_files = set(subjects)
        assert (
            n := len(from_splits - from_files)
        ) == 0, f"Missing {n} subjects listed in splits"
        assert (
            n := len(from_files - from_splits)
        ) == 0, f"Uncounted {n} subjects not listed in splits"

        subjects = []
        n_labels = None
        res, dataset, version, group, modality, axis = key
        res = int(remove_prefix(res, "res"))
        for (name, img), (segname, seg) in tqdm(
            zip(img_loader, seg_loader), total=len(imgs), leave=False
        ):
            assert name.split("/")[:-1] == segname.split("/")[:-1]
            if n_labels:
                assert (
                    n_labels == seg.shape[-1]
                )  # TODO: change to zero when Victor changes order
            else:
                n_labels = seg.shape[-1]
            subject = name.split("/")[-2]
            db[subject] = postprocess_pair(img, seg, res, n_labels)
            subjects.append(subject)

        slicing, revision = version.split("_")

        # TODO: FIXME
        attrs = dict(
            dataset=dataset,
            version="v4.1",
            slicing=slicing,
            group=group,
            modality=modality,
            axis=axis,
            resolution=res,
            n_labels=n_labels,
        )
        db["_subjects"] = subjects
        db["_samples"] = subjects
        db["_splits"] = splits
        db["_attrs"] = attrs
        db["_pop_lab_amount"] = get_label_info(root, key)
    return ThunderDataset(dst)


def reprocess_hierarchy(force,
                        max_workers,
                        suppress_exceptions,
                        dry_run,
                        VERSION):
    
    root = pathlib.Path("/home/vib9/src/MegaMedical/processed/processed_raw")
    path = root
    outdir = pathlib.Path("/home/vib9/src/MegaMedical/processed/lmdb_processed")
    # TODO: Get all images
    all_imgs = list(root.glob("**/img.npy"))
    # 
    unique = regroup_files(all_imgs)
    N = len(unique)

    print(
            f"Processing {N} datasets with {sum(map(len, unique.values()))} subjects under {str(path)}"
        )

    errors = []

    for i, (key, imgs) in enumerate(unique.items(), start=1):
        res, dataset, version, group, modality, axis = key
        
        slicing, _ = version.split("_")
        dst = (
            outdir
            / f"{VERSION}/{res}/{slicing}/{dataset}/{group}/{modality}/{axis}"
        )

        if force or not (dst / "data.mdb").exists():
            print(f'{i:03d}/{N:03d} - Processing... {"/".join(key)}')
            if dry_run:
                continue
            t = Timer()
            if dst.exists():
                shutil.rmtree(str(dst))
            tmpdst = dst.with_suffix(".tmp")
            tmpdst.mkdir(parents=True, exist_ok=True)
            try:
                with t("process"):
                    thunderify(
                        root,
                        key,
                        imgs,
                        tmpdst,
                        VERSION,
                        max_workers=max_workers,
                    )
                print(
                    f'{i:03d}/{N:03d} - Processed in {t.measurements["process"]:.2g}s'
                )
                tmpdst.rename(dst)
            except KeyboardInterrupt as e:
                import sys

                sys.exit(1)
            except Exception as e:
                if suppress_exceptions:
                    printc(
                        f"{i:03d}/{N:03d} - Error processing {'/'.join(key)}",
                        color="RED",
                    )
                    errors.append("/".join(key))
                else:
                    raise e
        else:
            print(f'{i:03d}/{N:03d} - Already processed {"/".join(key)}')

    if len(errors) > 0:
        print(f"\nEncountered {len(errors)} errors:")
        for err in errors:
            print(err)

