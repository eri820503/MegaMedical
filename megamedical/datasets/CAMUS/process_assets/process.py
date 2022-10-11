from tqdm.notebook import tqdm_notebook
import glob
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put

class CAMUS:

    def __init__(self):
        self.name = "CAMUS"
        self.dset_info = {
            "4CH_ED":{
                "main":"CAMUS",
                "image_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "label_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "modality_names": ["Ultrasound"],
                "planes": [0],
                "clip_args":None,
                "norm_scheme":None
            },
            "4CH_ES":{
                "main":"CAMUS",
                "image_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "label_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "modality_names": ["Ultrasound"],
                "planes": [0],
                "clip_args":None,
                "norm_scheme":None
            },
            "2CH_ED":{
                "main":"CAMUS",
                "image_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "label_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "modality_names": ["Ultrasound"],
                "planes": [0],
                "clip_args":None,
                "norm_scheme":None
            },
            "2CH_ES":{
                "main":"CAMUS",
                "image_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "label_root_dir": f"{paths['DATA']}/CAMUS/original_unzipped/retrieved_09_29_2022/subjects",
                "modality_names": ["Ultrasound"],
                "planes": [0],
                "clip_args":None,
                "norm_scheme":None
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
                        # Get rid of "train-" in front
                        patient_name = image[6:]
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, f"{patient_name}_{dset_name}.mhd")
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{patient_name}_{dset_name}_gt.mhd")
                        
                        assert os.path.isfile(im_dir), "Valid image dir required!"
                        assert os.path.isfile(label_dir), "Valid label dir required!"

                        if load_images:
                            loaded_image = io.imread(im_dir, plugin = 'simpleitk').squeeze()
                            loaded_label = io.imread(label_dir, plugin = 'simpleitk').squeeze()
                                              
                            assert not (loaded_label is None), "Invalid Label"
                            assert not (loaded_image is None), "Invalid Image"
                        else:
                            loaded_image = None
                            loaded_label = io.imread(label_dir, plugin = 'simpleitk').squeeze()

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
