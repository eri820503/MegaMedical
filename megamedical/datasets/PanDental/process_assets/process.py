from PIL import Image
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class PanDental:

    def __init__(self):
        self.name = "PanDental"
        self.dset_info = {
            "v1":{
                "main": "PanDental",
                "image_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v1/Images",
                "label_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v1/orig_masks",
                "modality_names":["XRay"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
            },
            "v2":{
                "main": "PanDental",
                "image_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v2/Images",
                "label_root_dir":f"{paths['DATA']}/PanDental/original_unzipped/v2/Segmentation1",
                "modality_names":["XRay"],
                "planes":[0],
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
        image_list = os.listdir(self.dset_info[subdset]["image_root_dir"])
        skip_list = ["106.png", "107.png", "109.png", "111.png", "112.png",
                         "113.png", "116.png", "19.png", "39.png", "64.png", "65.png",
                         "68.png", "70.png", "76.png", "78.png", "79.png", "98.png"]
        if subdset == "v2":
            for item in skip_list:
                image_list.remove(item)
        image_list = sorted(image_list)
        subj_dict, res_dict = proc.process_image_list(process_PanDental_image,
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
        
        
global process_PanDental_image
def process_PanDental_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed'] or put.is_processed_check(item):
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'])

            loaded_image = np.array(Image.open(im_dir).convert('L'))

            if item['subdset'] == "v1":
                loaded_label = np.array(Image.open(label_dir))[...,0]
            else:
                loaded_label = (np.array(Image.open(label_dir).convert('L')) > 0).astype(int)

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