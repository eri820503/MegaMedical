import numpy as np
import matplotlib.pyplot as plt
import os
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths

def show_processing(dataset_object, subdset, show_hists=False):
    assert isinstance(subdset, str), "Must be a string."
    if subdset == "all":
        dset_names = list(dataset_object.dset_info.keys())
    else:
        dset_names = [subdset]
    for dataset in dset_names:
        dataset_object.proc_func(dataset, 
                                 show_hists=show_hists,
                                 show_imgs=True,
                                 redo_processed=True)

      
def show_dataset(dataset_object, subdset, version, resolution=128, subjs_to_vis=10):
    assert isinstance(subdset, str), "Must be a string."
    
    # Get list of sub-datasets
    if subdset == "all":
        dset_names = list(dataset_object.dset_info.keys())
    else:
        dset_names = [subdset]
    
    # Get list of processed dirs (if they exist)
    subdsets_data_dirs = []
    for sdset in dset_names:
        try:
            sdset_info = dataset_object.dset_info[sdset]
            if version in ["megamedical_v3.1", "midslice_v3.1"]:
                dset_path = os.path.join(paths["ROOT"], "megamedical", sdset_info["main"], "processed", version, sdset)
            elif version == "v3.2-midslice":
                dset_path = os.path.join("/share/sablab/nfs02/users/gid-dalcaav/data/v3.2-midslice", sdset_info["main"], sdset)
            else:
                raise NotImplemented(f"Version of {sdset_info['main']}/{sdset} does not exist!")
            subdsets_data_dirs.append(dset_path)
        except Exception as e:
            print(e)
            continue
   
    # Display each of the subjects
    for sdd in subdsets_data_dirs:
        modalities = os.listdir(sdd)
        for mod in modalities:
            mod_dir = os.path.join(sdd, mod)
            subjs = os.listdir(mod_dir)
            chosen_subjs = np.random.choice(subjs, subjs_to_vis)
            for sub in subjs:
                img_dir = os.path.join(mod_dir, sub, f"img_{resolution}.npy")
                seg_dir = os.path.join(mod_dir, sub, f"seg_{resolution}.npy")
                img = np.load(img_dir)
                seg = np.load(seg_dir)
                for plane in range(3):
                    f, axarr = plt.subplots(nrows=1, ncols=(seg.shape[-1] + 1), figsize=[9,3])
                    img_slice = np.take(img, img.shape[plane]//2, plane)
                    seg_slice = np.take(seg, seg.shape[plane]//2, plane)
                    axarr[0].imshow(img_slice)
                    for l_idx in range(seg.shape[-1]):
                        bin_slice = seg_slice[..., l_idx]
                        axarr[l_idx + 1].imshow(bin_slice)
                    plt.show()
                
        
        
                             
        