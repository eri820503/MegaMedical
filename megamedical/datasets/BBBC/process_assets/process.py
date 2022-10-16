import cv2
from tqdm.notebook import tqdm_notebook
import glob
import os
import numpy as np

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class BBBC:

    def __init__(self):
        self.name = "BBBC"
        self.dset_info = {
            "BBBC003":{
                "main":"BBBC",
                "image_root_dir":f"{paths['DATA']}/BBBC/original_unzipped/BBBC003/mouse_embryos_dic_images",
                "label_root_dir":f"{paths['DATA']}/BBBC/original_unzipped/BBBC003/mouse_embryos_dic_foreground",
                "modality_names":["NA"],
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
        image_list = sorted(os.listdir(self.dset_info[subdset]["image_root_dir"]))
        subj_dict, res_dict = proc.process_image_list(process_BBBC_image,
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


global process_BBBC_image
def process_BBBC_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        if item['redo_processed']:
            rtp = put.check_proc_res(item)
        else:
            rtp = item["resolutions"]
        if len(rtp) > 0:
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'])

            assert os.path.isfile(im_dir), "Valid image dir required!"
            assert os.path.isfile(label_dir), "Valid label dir required!"

            if item['load_images']:
                loaded_image = cv2.imread(im_dir)
                loaded_label = cv2.imread(label_dir)

                loaded_image = np.array(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY))
                loaded_label = np.array(cv2.cvtColor(loaded_label, cv2.COLOR_BGR2GRAY))
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = cv2.imread(label_dir)
                loaded_label = np.array(cv2.cvtColor(loaded_label, cv2.COLOR_BGR2GRAY))

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