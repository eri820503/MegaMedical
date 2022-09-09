import cv2
from tqdm.notebook import tqdm_notebook
import glob
import numpy as np
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class LGGFlair:

    def __init__(self):
        self.name = "LGGFlair"
        self.dset_info = {
            "retrieved_2021_10_11":{
                "main":"LGGFlair",
                "image_root_dir":f"{paths['DATA']}/LGGFlair/original_unzipped/retrieved_2021_10_11",
                "label_root_dir":f"{paths['DATA']}/LGGFlair/original_unzipped/retrieved_2021_10_11",
                "modality_names":["Flair"],
                "planes":[2],
                "labels": [1,2,3],
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
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(self.name, dset_name, save, version, self.dset_info[dset_name])
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            try:
                proc_dir_template = os.path.join(proc_dir, f"midslice_v{version}", dset_name, "*", image)
                if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                    file_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                    image_file_list = []
                    label_file_list = []

                    num_slices = int(len(os.listdir(file_dir))/2)
                    for i in range(1, num_slices + 1):
                        image_file_list.append(os.path.join(file_dir,f"{image}_{str(i)}.tif"))
                        label_file_list.append(os.path.join(file_dir,f"{image}_{str(i)}_mask.tif"))

                    if load_images:
                        loaded_image = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in image_file_list], axis=2)
                        loaded_label = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in label_file_list], axis=2)
                        assert not (loaded_label is None), "Invalid Label"
                        assert not (loaded_image is None), "Invalid Image"
                    else:
                        loaded_image = None
                        loaded_label = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in label_file_list], axis=2)

                    proc_return = proc_func(proc_dir,
                                          version,
                                          dset_name,
                                          image, 
                                          loaded_image,
                                          loaded_label,
                                          self.dset_info[dset_name],
                                          show_hists=show_hists,
                                          show_imgs=show_imgs,
                                          save=save)

                    if accumulate:
                        accumulator.append(proc_return)
            except Exception as e:
                print(e)
                #raise ValueError
        if accumulate:
            return proc_dir, accumulator