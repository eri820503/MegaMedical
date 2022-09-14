import h5py
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

#New line!
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
                "labels": [1,2,3],
                "clip_args":None,
                "norm_scheme":"MR",
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
        hf = h5py.File(os.path.join(self.dset_info[dset_name]["image_root_dir"],'dataset.hdf5'), 'r')
  
        chosen_inds = np.sort(np.random.choice(np.arange(len(hf["image"])), 1000))
        images = hf["image"]
        segs = hf["mask"]
        accumulator = []
        for image in tqdm_notebook(chosen_inds, desc=f'Processing: {dset_name}'):
            try:
                proc_dir_template = os.path.join(proc_dir, f"midslice_v{version}", dset_name, "*", f"img{image}")
                if redo_processed or (len(glob.glob(proc_dir_template)) == 0):

                    if load_images:
                        loaded_image = np.array(images[image, ...])
                        loaded_label = np.array(segs[image, ...])
                        assert not (loaded_label is None), "Invalid Label"
                        assert not (loaded_image is None), "Invalid Image"
                    else:
                        loaded_image = None
                        loaded_label = np.array(segs[image, ...])

                    proc_return = proc_func(proc_dir,
                                              version,
                                              dset_name,
                                              str(image), 
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