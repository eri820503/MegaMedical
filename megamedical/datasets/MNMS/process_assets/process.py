import nibabel as nib
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class MNMS:

    def __init__(self):
        self.name = "MNMS"
        self.dset_info = {
            "2020":{
                "main":"MNMS",
                "image_root_dir":f"{paths['DATA']}/MNMS/original_unzipped/2020/OpenDataset",
                "label_root_dir":f"{paths['DATA']}/MNMS/original_unzipped/2020/OpenDataset",
                "modality_names":["T1"],
                "planes":[2],
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
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        proc_dir = os.path.join(paths['ROOT'], "processed")
        res_dict = {}
        for resolution in resolutions:
            accumulator = []
            for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
                try:
                    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                    proc_dir_template = os.path.join(proc_dir, f"res{resolution}", self.name, f"midslice_v{version}", dset_name, "*/*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{image}_sa.nii.gz")
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_sa_gt.nii.gz")

                        if load_images:
                            loaded_image = nib.load(im_dir)
                            loaded_label = nib.load(label_dir)

                            loaded_image = loaded_image.get_fdata()
                            loaded_label = loaded_label.get_fdata()

                            # What is this? Why 0?
                            loaded_image = loaded_image[...,0]
                            loaded_label = loaded_label[...,0]
                            assert not (loaded_label is None), "Invalid Label"
                            assert not (loaded_image is None), "Invalid Image"
                        else:
                            loaded_image = None
                            loaded_label = nib.load(label_dir).get_fdata()
                            loaded_label = loaded_label[...,0]

                        proc_return = proc_func(proc_dir,
                                              version,
                                              dset_name,
                                              image, 
                                              loaded_image,
                                              loaded_label,
                                              self.dset_info[dset_name],
                                              show_hists=show_hists,
                                              show_imgs=show_imgs,
                                              res=resolution,
                                              save=save)

                        if accumulate:
                            accumulator.append(proc_return)
                except Exception as e:
                    print(e)
                    #raise ValueError
            res_dict[resolution] = accumulator
        if accumulate:
            return proc_dir, res_dict