import pathlib
from tqdm.auto import tqdm

from pylot.util.thunder import ThunderReader
from pylot.util.ioutil import autosave


def extract(source: pathlib.Path, destination: pathlib.Path):
    reader = ThunderReader(source)
    destination = pathlib.Path(destination)
    destination.mkdir(exist_ok=True, parents=True)

    for key, value in tqdm(reader.items(), total=len(reader), leave=False):
        if key.startswith("_"):
            autosave(value, destination / f"{key}.json")
        else:
            img, seg = value
            autosave(img, destination / f"{key}/img.npy")
            autosave(seg, destination / f"{key}/seg.npy")


if __name__ == "__main__":
    import typer

    typer.run(extract)
