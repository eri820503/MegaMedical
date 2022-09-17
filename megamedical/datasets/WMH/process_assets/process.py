import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import numpy as np
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class WMH:

    def __init__(self):
        self.name = "WMH"
        self.dset_info = {
            "retrieved_2022_03_10":{
                "main":"WMH",
                "image_root_dir":f"{paths['DATA']}/WMH/original_unzipped/retrieved_2022_03_10/public",
                "label_root_dir":f"{paths['DATA']}/WMH/original_unzipped/retrieved_2022_03_10/public",
                "modality_names":["FLAIR"],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
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
        proc_dir = pps.make_processed_dir(self.name, dset_name, save, version, self.dset_info[dset_name])
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            try:
                proc_dir_template = os.path.join(proc_dir, f"midslice_v{version}", dset_name, "*", image)
                if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                    im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "pre/FLAIR.nii.gz")
                    #T1_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "pre/T1.nii.gz")
                    label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, "wmh.nii.gz")

                    #flair = np.array(nib.load(FLAIR_dir).dataobj)
                    #t1 = np.array(nib.load(T1_dir).dataobj)

                    if load_images:
                        loaded_image = put.resample_nib(nib.load(im_dir))
                        loaded_label = put.resample_mask_to(nib.load(label_dir), loaded_image)

                        loaded_image = loaded_image.get_fdata()
                        loaded_label = loaded_label.get_fdata()
                        assert not (loaded_label is None), "Invalid Label"
                        assert not (loaded_image is None), "Invalid Image"
                    else:
                        loaded_image = None
                        loaded_label = nib.load(label_dir).get_fdata()

                    proc_return = proc_func(proc_dir,
                                              version,
                                              dset_name,
                                              image, 
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
