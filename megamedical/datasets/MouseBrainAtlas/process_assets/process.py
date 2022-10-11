import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import numpy as np
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put

class MouseBrainAtlas:

    def __init__(self):
        self.name = "MouseBrainAtlas"
        self.dset_info = {
            "FVB_NCrl":{
                "main":"MouseBrainAtlas",
                "image_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "label_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "modality_names": ["MRI"],
                "planes": [2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "NeAt":{
                "main":"MouseBrainAtlas",
                "image_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "label_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "modality_names": ["MRI"],
                "planes": [2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "Tc1_Cerebellum":{
                "main":"MouseBrainAtlas",
                "image_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "label_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "modality_names": ["MRI"],
                "planes": [2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "rTg4510":{
                "main":"MouseBrainAtlas",
                "image_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "label_root_dir": f"{paths['DATA']}/MouseBrainAtlas/original_unzipped/retrieved_09_05_2022/mouse-brain-atlas-master",
                "modality_names": ["MRI"],
                "planes": [2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
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
        image_list = sorted(os.listdir(self.dset_info[dset_name]["image_root_dir"]))
        proc_dir = os.path.join(paths['ROOT'], "processed")
        res_dict = {}
        subj_dict = {}
        for resolution in resolutions:
            accumulator = []
            subj_accumulator = []
            for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
                try:
                    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                    template_root = os.path.join(proc_dir, f"res{resolution}", self.name)
                    mid_proc_dir_template = os.path.join(template_root, f"midslice_v{version}", dset_name, "*/*", image)
                    max_proc_dir_template = os.path.join(template_root, f"maxslice_v{version}", dset_name, "*/*", image)
                    if redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{image}_frame01.nii.gz")
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_frame01_gt.nii.gz")

                        assert os.path.isfile(im_dir), "Valid image dir required!"
                        assert os.path.isfile(label_dir), "Valid label dir required!"

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

                        # Set the name to be saved
                        subj_name = image.split(".")[0]
                        proc_return = proc_func(proc_dir,
                                                version,
                                                dset_name,
                                                subj_name, 
                                                loaded_image,
                                                loaded_label,
                                                self.dset_info[dset_name],
                                                show_hists=show_hists,
                                                show_imgs=show_imgs,
                                                res=resolution,
                                                save=save)
                        
                        if accumulate:
                            accumulator.append(proc_return)
                            subj_accumulator.append(subj_name)
                except Exception as e:
                    print(e)
                    #raise ValueError
            res_dict[resolution] = accumulator
            subj_dict[resolution] = subj_accumulator
        if accumulate:
            return proc_dir, subj_dict, res_dict
