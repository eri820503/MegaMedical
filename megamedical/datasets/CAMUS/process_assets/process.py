from tqdm.notebook import tqdm_notebook
import glob
import os
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

from megamedical.src import processing as proc
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
        subj_dict, res_dict = proc.process_image_list(process_CAMUS_image,
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
        
        
global process_CAMUS_image
def process_CAMUS_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or put.is_processed_check(item):
            # Get rid of "train-" in front
            patient_name = item['image'][6:]
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'], f"{patient_name}_{item['subdset']}.mhd")
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'], f"{patient_name}_{item['subdset']}_gt.mhd")

            assert os.path.isfile(im_dir), "Valid image dir required!"
            assert os.path.isfile(label_dir), "Valid label dir required!"

            if item['load_images']:
                loaded_image = io.imread(im_dir, plugin = 'simpleitk').squeeze()
                loaded_label = io.imread(label_dir, plugin = 'simpleitk').squeeze()

                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = io.imread(label_dir, plugin = 'simpleitk').squeeze()

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
                                        res=item['resolution'],
                                        save=item['save'])

            return proc_return, subj_name
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None