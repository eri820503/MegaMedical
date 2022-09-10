import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import os
import numpy as np

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class BrainDevelopment:

    def __init__(self):
        self.name = "BrainDevelopment"
        self.dset_info = {
            "HammersAtlasDatabase":{
                "main":"BrainDevelopment",
                "image_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/HammersAtlasDatabase/Hammers67n20/images",
                "label_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/HammersAtlasDatabase/Hammers67n20/segs",
                "modality_names":["T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "PediatricAtlas":{
                "main":"BrainDevelopment",
                "image_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/PediatricAtlas/images",
                "label_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/PediatricAtlas/segmentations",
                "modality_names":["T1"],
                "planes":[0, 1, 2],
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
                    im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                    seg_addon = "-seg.nii.gz" if dset_name == "HammersAtlasDatabase" else "_seg_83ROI.nii.gz"
                    label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image.replace(".nii.gz", seg_addon))

                    assert os.path.isfile(im_dir), "Valid image dir required!"
                    assert os.path.isfile(label_dir), "Valid label dir required!"

                    if load_images:
                        loaded_image = nib.load(im_dir).get_fdata().squeeze()
                        loaded_label = nib.load(label_dir).get_fdata().squeeze()
                        assert not (loaded_label is None), "Invalid Label"
                        assert not (loaded_image is None), "Invalid Image"
                    else:
                        loaded_image = None
                        loaded_label = nib.load(label_dir).get_fdata().squeeze()

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