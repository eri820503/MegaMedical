import h5py
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class TUCC:

    def __init__(self):
        self.name = "TUCC"
        self.dset_info = {
            "retreived_2022_03_06":{
                "main": "TUCC",
                "image_root_dir":f"{paths['DATA']}/TUCC/original_unzipped/retreived_2022_03_04",
                "label_root_dir":f"{paths['DATA']}/TUCC/original_unzipped/retreived_2022_03_04",
                "modality_names":["Ultrasound"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
            }
        }

    def proc_func(self,
                  subdset,
                  pps_function,
                  parallelize=False,
                  load_images=True,
                  accumulate=False,
                  version=None,
                  show_imgs=False,
                  save=False,
                  show_hists=False,
                  resolutions=None,
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert subdset in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = os.path.join(paths['ROOT'], "processed")
        # For reproducibility, fix seed
        np.random.seed(0)
        hf = h5py.File(os.path.join(self.dset_info[dset_name]["image_root_dir"],'dataset.hdf5'), 'r')
        chosen_inds = sorted(np.sort(np.random.choice(np.arange(len(hf["image"])), 1000)))
        images = hf["image"]
        segs = hf["mask"]
        subj_dict, res_dict = proc.process_image_list(process_AMOS_image,
                                                      proc_dir,
                                                      image_list,
                                                      parallelize,
                                                      pps_function,
                                                      resolutions,
                                                      self.name,
                                                      subdset,
                                                      self.dset_info,
                                                      redo_processed,
                                                      load_images,
                                                      show_hists,
                                                      version,
                                                      show_imgs,
                                                      accumulate,
                                                      save)
        if accumulate:
            return proc_dir, subj_dict, res_dict
        
        
global process_TUCC_image
def process_TUCC_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or is_processed_check(item):
            if load_images:
                loaded_image = np.array(images[image, ...])
                loaded_label = np.array(segs[image, ...])
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = np.array(segs[image, ...])

            # Set the name to be saved
            subj_name = f"frame_{image}"
            pps_function = item['pps_function']
            proc_return = pps_function(item['proc_dir'],
                                        item['version'],
                                        item['subdset'],
                                        subj_name, 
                                        loaded_image,
                                        loaded_label,
                                        dset_info[item['subdset']],
                                        show_hists=item['show_hists'],
                                        show_imgs=item['show_imgs'],
                                        res=item['resolution'],
                                        save=item['save'])

            return proc_return, subj_name
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None