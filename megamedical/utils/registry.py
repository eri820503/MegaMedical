import os

paths = {
    "ROOT": "/data/ddmg/users/vbutoi/projects/MegaMedical",
    "DATA": "/storage/megamedical/v4-raw/raw_data",
    "PROC": "/storage/megamedical/v4-raw/processed"
}

list_of_datasets = os.listdir(paths["DATA"])
dataset_registry = {dataset : f"megamedical.datasets.{dataset}.process_assets" for dataset in list_of_datasets}
