import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
from PIL import Image
import numpy as np
import json
from pycocotools import mask
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class CheXplanation:

    def __init__(self):
        self.name = "CheXplanation"
        self.dset_info = {
            "retreived_2022_03_04":{
                "main":"Chexplanation",
                "image_root_dir":f"{paths['DATA']}/CheXplanation/original_unzipped/v1.0/CheXplanation/CheXpert-v1.0/valid",
                "label_root_dir":f"{paths['DATA']}/CheXplanation/original_unzipped/v1.0/CheXplanation",
                "modality_names":["XRay"],
                "planes":[0],
                "clip_args": None,
                "norm_scheme": None,
                "do_clip": False,
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
        proc_dir = os.path.join(paths['ROOT'], "processed")
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], "gt_segmentation_val.json")
        label_file = open(label_dir)
        label_json = json.load(label_file)
        proc_dir = os.path.join(paths['ROOT'], "processed")
        res_dict = {}
        for resolution in resolutions:
            accumulator = []
            for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
                try:
                    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                    proc_dir_template = os.path.join(proc_dir, f"res{resolution}", self.name, f"midslice_v{version}", dset_name, "*/*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "study1/view1_frontal.jpg")
                        assert os.path.isfile(im_dir), "Valid image dir required!"

                        if load_images:
                            loaded_image = np.array(Image.open(im_dir).convert('L'))
                            loaded_labels = []
                            subj_dict = label_json[f"{image}_study1_view1_frontal"]
                            for label in subj_dict.keys():
                                loaded_labels.append(mask.decode(subj_dict[label]))
                            loaded_label = np.argmax(np.stack(loaded_labels), axis=0)
                            assert not (loaded_label is None), "Invalid Label"
                            assert not (loaded_image is None), "Invalid Image"
                        else:
                            loaded_image = None
                            loaded_labels = []
                            subj_dict = label_json[f"{image}_study1_view1_frontal"]
                            for label in subj_dict.keys():
                                loaded_labels.append(mask.decode(subj_dict[label]))
                            loaded_label = np.argmax(np.stack(loaded_labels), axis=0)

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