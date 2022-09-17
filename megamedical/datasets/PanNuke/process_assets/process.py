import nibabel as nib
from tqdm.notebook import tqdm_notebook
from PIL import Image
import numpy as np
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class PanNuke:

    def __init__(self):
        self.name = "PanNuke"
        self.dset_info = {
            "Fold1":{
                "main":"PanNuke",
                "image_root_dir":f"{paths['DATA']}/PanNuke/original_unzipped/retreived_2022_03_04/Fold1/images/fold1/images.npy",
                "label_root_dir":f"{paths['DATA']}/PanNuke/original_unzipped/retreived_2022_03_04/Fold1/masks/fold1/masks.npy",
                "modality_names":["NA"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
            "Fold2":{
                "main":"PanNuke",
                "image_root_dir":f"{paths['DATA']}/PanNuke/original_unzipped/retreived_2022_03_04/Fold2/images/fold2/images.npy",
                "label_root_dir":f"{paths['DATA']}/PanNuke/original_unzipped/retreived_2022_03_04/Fold2/masks/fold2/masks.npy",
                "modality_names":["NA"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
            "Fold3":{
                "main":"PanNuke",
                "image_root_dir":f"{paths['DATA']}/PanNuke/original_unzipped/retreived_2022_03_04/Fold3/images/fold3/images.npy",
                "label_root_dir":f"{paths['DATA']}/PanNuke/original_unzipped/retreived_2022_03_04/Fold3/masks/fold3/masks.npy",
                "modality_names":["NA"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            }
        }

    def proc_func(self,
                  dset_name,
                  proc_func,
                  load_images=True,
                  accumulate=False,
                  version=None,
                  show_imgs=False,
                  save=False,
                  show_hists=False,
                  resolutions=None,
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = os.path.join(paths['DATA'], self.name, "processed")
        
        volumes_array = np.load(self.dset_info[dset_name]["image_root_dir"])
        labels_array = np.load(self.dset_info[dset_name]["label_root_dir"])
        
        image_list = list(range(volumes_array.shape[0]))
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            try:
                proc_dir_template = os.path.join(proc_dir, f"midslice_v{version}", dset_name, "*", str(image))
                if redo_processed or (len(glob.glob(proc_dir_template)) == 0):

                    # "clever" hack to get Image.fromarray to work
                    loaded_image = volumes_array[image,...]
                    loaded_image = 0.2989*loaded_image[...,0] + 0.5870*loaded_image[...,1] + 0.1140*loaded_image[...,2] 

                    loaded_label = np.transpose(labels_array[image,...], (2, 0, 1))
                    background_label = np.zeros((1, loaded_label.shape[1], loaded_label.shape[2]))
                    loaded_label = np.concatenate([background_label, loaded_label], axis=0)
                    loaded_label = np.argmax(loaded_label, axis=0)

                    assert not (loaded_image is None), "Invalid Image"
                    assert not (loaded_label is None), "Invalid Label"

                    proc_return = proc_func(proc_dir,
                                            version,
                                            dset_name,
                                            str(image), 
                                            loaded_image,
                                            loaded_label,
                                            self.dset_info[dset_name],
                                            show_hists=show_hists,
                                            show_imgs=show_imgs,
                                            resolutions=resolutions,
                                            save=save)

                    if accumulate:
                        accumulator.append(proc_return)
            except Exception as e:
                print(e)
                #raise ValueError
        if accumulate:
            return proc_dir, accumulator
