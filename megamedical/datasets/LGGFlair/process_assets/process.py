import cv2
from tqdm.notebook import tqdm_notebook
import glob
import numpy as np
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class LGGFlair:

    def __init__(self):
        self.name = "LGGFlair"
        self.dset_info = {
            "retrieved_2021_10_11":{
                "main":"LGGFlair",
                "image_root_dir":f"{paths['DATA']}/LGGFlair/original_unzipped/retrieved_2021_10_11",
                "label_root_dir":f"{paths['DATA']}/LGGFlair/original_unzipped/retrieved_2021_10_11",
                "modality_names":["Flair"],
                "planes":[2],
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
        subj_dict, res_dict = proc.process_image_list(process_LGGFlair_image,
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
        
        
global process_LGGFlair_image
def process_LGGFlair_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed']:
            rtp = put.check_proc_res(item)
        else:
            rtp = item["resolutions"]
        if len(rtp) > 0:
            file_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            image_file_list = []
            label_file_list = []

            num_slices = int(len(os.listdir(file_dir))/2)
            for i in range(1, num_slices + 1):
                image_file_list.append(os.path.join(file_dir,f"{item['image']}_{str(i)}.tif"))
                label_file_list.append(os.path.join(file_dir,f"{item['image']}_{str(i)}_mask.tif"))

            if item['load_images']:
                loaded_image = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in image_file_list], axis=2)
                loaded_label = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in label_file_list], axis=2)
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = np.concatenate([cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)[...,np.newaxis] for f in label_file_list], axis=2)

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
                                        resolutions=rtp,
                                        save=item['save'])

            return proc_return, subj_name
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None