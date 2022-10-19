import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import os
import numpy as np

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class BrainDevelopment:

    def __init__(self):
        self.name = "BrainDevelopment"
        self.dset_info = {
            "HammersAtlasDatabase":{
                "main":"BrainDevelopment",
                "image_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/HammersAtlasDatabase/Hammers67n20/images",
                "label_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/HammersAtlasDatabase/Hammers67n20/segs",
                "modality_names":["T1"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
            },
            "PediatricAtlas":{
                "main":"BrainDevelopment",
                "image_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/PediatricAtlas/images",
                "label_root_dir":f"{paths['DATA']}/BrainDevelopment/original_unzipped/PediatricAtlas/segmentations",
                "modality_names":["T1"],
                "planes":[0, 1, 2],
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
        subj_dict, res_dict = proc.process_image_list(process_BrainDevelopment_image,
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
        
        
global process_BrainDevelopment_image
def process_BrainDevelopment_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        rtp = item["resolutions"] if item['redo_processed'] else put.check_proc_res(item)
        if len(rtp) > 0:
            im_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            seg_addon = "-seg.nii.gz" if item['subdset'] == "HammersAtlasDatabase" else "_seg_83ROI.nii.gz"
            label_dir = os.path.join(dset_info[item['subdset']]["label_root_dir"], item['image'].replace(".nii.gz", seg_addon))

            assert os.path.isfile(im_dir), "Valid image dir required!"
            assert os.path.isfile(label_dir), "Valid label dir required!"

            if item['load_images']:
                loaded_image = nib.load(im_dir).get_fdata().squeeze()
                loaded_label = nib.load(label_dir).get_fdata().squeeze()
                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = nib.load(label_dir).get_fdata().squeeze()
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