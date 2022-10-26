import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
from PIL import Image
import numpy as np
import json
from pycocotools import mask
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class CheXplanation:

    def __init__(self):
        self.name = "CheXplanation"
        self.dset_info = {
            "retreived_2022_03_04":{
                "main":"CheXplanation",
                "image_root_dir":f"{paths['DATA']}/CheXplanation/original_unzipped/v1.0/CheXplanation/CheXpert-v1.0/valid",
                "label_root_dir":f"{paths['DATA']}/CheXplanation/original_unzipped/v1.0/CheXplanation",
                "modality_names":["XRay"],
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
        subj_dict, res_dict = proc.process_image_list(process_CheXplanation_image,
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
        
        
global process_CheXplanation_image
def process_CheXplanation_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        file_name = item['image']
        item['image'] = file_name.split(".")[0]
        rtp = item["resolutions"] if item['redo_processed'] else put.check_proc_res(item)
        if len(rtp) > 0:
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], file_name, "study1/view1_frontal.jpg")
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], "gt_segmentation_val.json")
            assert os.path.isfile(im_dir), "Valid image dir required!"
            assert os.path.isfile(label_dir), "Valid label dir required!"
            
            label_file = open(label_dir)
            label_json = json.load(label_file)

            if item['load_images']:
                loaded_image = np.array(Image.open(im_dir).convert('L'))
                loaded_labels = []
                subj_dict = label_json[f"{file_name}_study1_view1_frontal"]
                for label in subj_dict.keys():
                    loaded_labels.append(mask.decode(subj_dict[label]))
                loaded_label = np.argmax(np.stack(loaded_labels), axis=0)
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_labels = []
                subj_dict = label_json[f"{file_name}_study1_view1_frontal"]
                for label in subj_dict.keys():
                    loaded_labels.append(mask.decode(subj_dict[label]))
                loaded_label = np.argmax(np.stack(loaded_labels), axis=0)
                
            label_file.close()

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
    except Exception as e:
        print(e)
        return None, None