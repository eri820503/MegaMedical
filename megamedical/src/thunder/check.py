import itertools
import pathlib

from pydantic import validate_arguments
from tqdm import tqdm
import einops as E

from pylot.util import ThunderReader
from pylot.util.future import remove_prefix


@validate_arguments
def check_dataset(path: pathlib.Path):
    assert path.exists()
    db = ThunderReader(path)
    s = ""
    subjects = db["_subjects"]
    splits = db["_splits"]
    a = db["_attrs"]
    res = a["resolution"]
    n_labels = a["n_labels"]
    k = subjects[0]
    img, seg = db[k]
    if not img.shape in [(res, res), (n_labels, res, res)]:
        s += "\tImage shape mismatch\n"
    if not seg.shape == (n_labels, res, res):
        s += "\tSegmentation shape mismatch\n"

    from_splits = set(sum(splits.values(), start=[]))
    from_files = set(subjects)
    if (n := len(from_splits - from_files)) != 0:
        missing = list(from_splits - from_files)
        s += f"\tMissing {n} subjects listed in splits\n\t\t{missing}"
    if (n := len(from_files - from_splits)) != 0:
        missing = list(from_files - from_splits)
        s += f"\tUnaccounted {n} subjects not listed in splits\n\t\t{missing}"

    return s.strip('\n')


@validate_arguments
def all_paths(root: pathlib.Path):

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


if __name__ == "__main__":

    root = pathlib.Path("/storage/megamedical/v4.0")
    paths = all_paths(root)
    N = len(paths)
    for i, p in enumerate(paths, start=1):
        if err := check_dataset(p):
            relp = remove_prefix(str(p), str(root))
            print(relp)
            print(err)
