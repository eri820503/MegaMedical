import nibabel as nib
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put

class TotalSeg:

    def __init__(self):
        self.name = "TotalSeg"
        self.dset_info = {
            "retreived_09_01_2022":{
                "main":"TotalSeg",
                "image_root_dir": f"{paths['DATA']}/TotalSeg/original_unzipped/retreived_09_01_2022/Totalsegmentator_dataset",
                "label_root_dir": f"{paths['DATA']}/TotalSeg/original_unzipped/retreived_09_01_2022/Totalsegmentator_dataset",
                "modality_names": ["CT"],
                "planes": [0, 1, 2],
                "labels": [1,2,3],
                "clip_args": [-500, 1000],
                "norm_scheme":"CT",
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
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        proc_dir = pps.make_processed_dir(self.name, dset_name, save, version, self.dset_info[dset_name])
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            try:
                proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                    im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "ct.nii.gz")
                    segementation_folder = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, "segmentations")

                    assert os.path.isfile(im_dir), "Valid image dir required!"

                    if load_images:
                        loaded_image = nib.load(im_dir).get_fdata()
                        labels = [nib.load(os.path.join(segementation_folder, target)).get_fdata() for target in os.listdir(segementation_folder)]
                        background = np.zeros(loaded_image.shape)
                        labels.append(background)
                        combined_labels = np.stack(labels)
                        loaded_label = np.argmax(combined_labels, axis=0)
                        
                        assert not (loaded_label is None), "Invalid Label"
                        assert not (loaded_image is None), "Invalid Image"
                    else:
                        loaded_image = None
                        labels = [nib.load(os.path.join(segementation_folder, target)).get_fdata() for target in os.listdir(segementation_folder)]
                        background = np.zeros(loaded_image.shape)
                        labels.append(background)
                        loaded_label = np.argmax(np.concatenate(labels))

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
