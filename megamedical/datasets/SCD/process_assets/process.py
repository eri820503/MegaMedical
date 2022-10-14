import imageio as io
import nrrd
import numpy as np
from tqdm.notebook import tqdm_notebook
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class SCD:

    def __init__(self):
        self.name = "SCD"
        self.dset_info = {
            "LAS":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAS/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAS/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "LAF_Pre":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Pre/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Pre/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "LAF_Post":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Post/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Post/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "VIS_pig":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_pig/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_pig/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "VIS_human":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_human/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_human/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
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
        image_list = sorted(os.listdir(self.dset_info[subdset]["image_root_dir"]))
        subj_dict, res_dict = proc.process_image_list(process_SCD_image,
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
        
        
global process_SCD_image
def process_SCD_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or put.is_processed_check(item):
            if item['subdset']=="LAS":
                im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], "image.mhd")
                label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], "gt_binary.mhd")
                loaded_image = io.imread(im_dir, plugin='simpleitk')
                loaded_label = io.imread(label_dir, plugin='simpleitk')
            else:
                if item['subdset']=="LAF_Pre":
                    im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image']) + f"/de_a_{item['image'][1:]}.nrrd"
                    label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image']) + f"/la_seg_a_{item['image'][1:]}.nrrd"
                elif item['subdset']=="LAF_Post":
                    im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image']) + f"/de_b_{item['image'][1:]}.nrrd"
                    label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image']) + f"/la_seg_b_{item['image'][1:]}.nrrd"
                elif item['subdset'] in ["VIS_pig","VIS_human"]:
                    im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image']) + "/" + item['image'] + "_de.nrrd"
                    label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image']) + "/" + item['image'] + "_myo.nrrd"
                loaded_image, _ = nrrd.read(im_dir)
                loaded_label, _ = nrrd.read(label_dir)

            assert not (loaded_image is None), "Invalid Image"
            assert not (loaded_label is None), "Invalid Label"

            # Set the name to be saved
            subj_name = item['image'].split(".")[0]
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
                                        resolutions=item['resolutions'],
                                        save=item['save'])

            return proc_return, subj_name
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None