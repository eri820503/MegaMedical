import os

paths = {
    "ROOT": "/home/vib9/src/MegaMedical",
    "DATA": "/home/vib9/src/MegaMedical/megamedical/datasets",
    "PROC": "/home/vib9/src/MegaMedical/processed"
}

list_of_datasets = os.listdir(paths["DATA"])
dataset_registry = {dataset : f"megamedical.datasets.{dataset}.process_assets" for dataset in list_of_datasets}