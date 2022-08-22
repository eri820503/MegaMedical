import seaborn as sns
import os
from tqdm.notebook import tqdm_notebook
import numpy as np
from megamedical.utils.registry import paths
import megamedical.utils as utils

def vis_label_hists(dsets, nbins=100):
    dataset_objects = [utils.build_dataset(ds) for ds in dsets]
    for do in dataset_objects:
        processed_dir = os.path.join(paths["DATA"], do.name,
                                     "processed/megamedical_v4.0")
        subdsets = os.listdir(processed_dir)
        label_dir = os.path.join(processed_dir, subdsets[0])
        modalities = os.listdir(label_dir)
        subject_dir = os.path.join(label_dir, modalities[0])

        label_amounts_dicts = [{} for _ in range(len(do.dset_info[subdsets[0]]["planes"]))]
        for subj in tqdm_notebook(os.listdir(subject_dir)):
            seg_dir = os.path.join(subject_dir, subj, "img_128.npy")
            seg = np.load(seg_dir)
            for pidx, plane in enumerate(do.dset_info[subdsets[0]]["planes"]):
                seg_slice = np.take(seg, seg.shape[plane]//2, plane)
                uni_lab = np.unique(seg_slice)
                for la in uni_lab:
                    if la in label_amounts_dicts[pidx].keys():
                        label_amounts_dicts[pidx][la] += 1
                    else:
                        label_amounts_dicts[pidx][la] = 1
                    print(label_amounts_dicts)

            

        
        

