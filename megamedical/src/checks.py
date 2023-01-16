import os
from pylot.util import autoload
from megamedical.utils.registry import paths

def verify_dataset(data_object,
                   subdset):
    split_root = os.path.join(paths["PROC"], "split_files")
    skip_datasets = ["COCA", "DDR", "MS_CMR", "EPISURG", "EchoNet", "HRF", "MNMS", "MouseBrainAtlas", "RibSeg", "SMIR", "SegThy", "TCIA", "TeethSeg", "TotalSegmentor"]
    skip_subdsets = ["ABIDE", "GSP", "VerSe20"]
    if data_object.name not in skip_datasets and subdset not in skip_subdsets:
        split_files = get_splits(split_root, data_object.name, subdset)
        from_splits = set(sum(split_files.values(), start=[]))
        for data_type in ["midslice_v4.0", "maxslice_v4.0"]:
            for res_dir in ["res64", "res128", "res256"]:
                for modality in data_object.dset_info[subdset]["modality_names"]:
                    for plane in data_object.dset_info[subdset]["planes"]:
                        proc_dir = os.path.join(paths["PROC"], res_dir, data_object.name, data_type, subdset, modality, str(plane))
                        from_files = set(os.listdir(proc_dir))
                        missing = list(from_splits - from_files)
                        if len(missing) > 0:
                            print(proc_dir)

def get_splits(root, dataset, group):
    return {
        split: autoload(os.path.join(root, f"{dataset}__{group}__{split}.txt"))
        for split in ("train", "val", "test")
    }
