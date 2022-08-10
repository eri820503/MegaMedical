import importlib
from megamedical.utils.registry import dataset_registry


def build_dataset(dataset):
    obj = getattr(importlib.import_module(dataset_registry[dataset]), dataset)()
    return obj