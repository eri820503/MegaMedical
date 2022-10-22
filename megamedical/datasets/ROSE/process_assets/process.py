from PIL import Image
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class ROSE:

    def __init__(self):
        self.name = "ROSE"
        self.dset_info = {
            "ROSE-1-DVC":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-1-DVC/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-1-DVC/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
            },
            "ROSE-1-SVC":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-1-SVC/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-1-SVC/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
            },
            "ROSE-1-SVC_DVC":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-1-SVC_DVC/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-1-SVC_DVC/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
            },
            "ROSE-2":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-2/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/original_unzipped/ROSE-2/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None
            }
        }

    def proc_func(self,
                  subdset,
                  task,
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
        subj_dict, res_dict = proc.process_image_list(process_ROSE_image,
                                                      proc_dir,
                                                      task,
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
        
        
global process_ROSE_image
def process_ROSE_image(item):
    dset_info = item['dset_info']
    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
    file_name = item['image']
    item['image'] = file_name.split(".")[0]
    rtp = item["resolutions"] if item['redo_processed'] else put.check_proc_res(item)
    if len(rtp) > 0:
        im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], file_name)
        if item['subdset'] in ["ROSE-1-DVC", "ROSE-1-SVC", "ROSE-2"]:
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], file_name)
        else:
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], file_name.replace(".png",".tif"))

        if item["load_images"]:
            loaded_image = np.array(Image.open(im_dir).convert('L'))
            loaded_label = np.array(Image.open(label_dir))
            assert not (loaded_label is None), "Invalid Label"
            assert not (loaded_image is None), "Invalid Image"
        else:
            loaded_image = None
            loaded_label = np.array(Image.open(label_dir))

        # Set the name to be saved
        subj_name = item['image']
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