from PIL import Image
import scipy.io
from tqdm.notebook import tqdm_notebook
import glob
import numpy as np
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class CoNSeP:

    def __init__(self):
        self.name = "CoNSeP"
        self.dset_info = {
            "retreived_2022_03_04":{
                "main":"CoNSeP",
                "image_root_dir":f"{paths['DATA']}/CoNSeP/original_unzipped/retreived_2022_03_04/CoNSeP/Train/Images",
                "label_root_dir":f"{paths['DATA']}/CoNSeP/original_unzipped/retreived_2022_03_04/CoNSeP/Train/Labels",
                "modality_names":["NA"],
                "planes":[0],
                "clip_args": None,
                "norm_scheme": None
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
        subj_dict, res_dict = proc.process_image_list(process_CoNSeP_image,
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

        
global process_CoNSeP_image
def process_CoNSeP_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        rtp = item["resolutions"] if item['redo_processed'] else put.check_proc_res(item)
        if len(rtp) > 0:
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            lab_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], f"{item['image'][:-4]}.mat")

            if item['load_images']:
                loaded_image = np.array(Image.open(im_dir).convert('L'))
                loaded_label = np.array(scipy.io.loadmat(lab_dir)["type_map"])
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = np.array(scipy.io.loadmat(lab_dir)["type_map"])

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