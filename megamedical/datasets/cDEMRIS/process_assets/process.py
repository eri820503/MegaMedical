import nibabel as nib
from tqdm.notebook import tqdm_notebook
import nrrd
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put

class cDEMRIS:

    def __init__(self):
        self.name = "cDEMRIS"
        self.dset_info = {
            "ISBI_2012_pre":{
                "main":"cDEMRIS",
                "image_root_dir": f"{paths['DATA']}/cDEMRIS/original_unzipped/ISBI_2012/pre",
                "label_root_dir": f"{paths['DATA']}/cDEMRIS/original_unzipped/ISBI_2012/pre",
                "modality_names": ["MRI"],
                "planes": [2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "ISBI_2012_post":{
                "main":"cDEMRIS",
                "image_root_dir": f"{paths['DATA']}/cDEMRIS/original_unzipped/ISBI_2012/post",
                "label_root_dir": f"{paths['DATA']}/cDEMRIS/original_unzipped/ISBI_2012/post",
                "modality_names": ["MRI"],
                "planes": [2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
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
        subj_dict, res_dict = proc.process_image_list(process_cDEMRIS_image,
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

    
global process_cDEMRIS_image
def process_cDEMRIS_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        file_name = item['image']
        item['image'] = file_name.split(".")[0]
        rtp = item["resolutions"] if item['redo_processed'] else put.check_proc_res(item)
        if len(rtp) > 0:
            vers = "a" if item['subdset'] == "ISBI_2012_pre" else "b"
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], file_name, f"de_{vers}_{file_name[1:]}.nrrd")
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], file_name, f"la_seg_{vers}_{file_name[1:]}.nrrd")

            assert os.path.isfile(im_dir), "Valid image dir required!"
            assert os.path.isfile(label_dir), "Valid label dir required!"

            if item['load_images']:
                loaded_image, _ = nrrd.read(im_dir)
                loaded_label, _ = nrrd.read(label_dir)
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label, _ = nrrd.read(label_dir)

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