from PIL import Image
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

from megamedical.src import processing as proc
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class BrainMetShare:

    def __init__(self):
        self.name = "BrainMetShare"
        self.dset_info = {
            "retrieved_2022_03_04":{
                "main": "BrainMetShare",
                "image_root_dir":f"{paths['DATA']}/BrainMetShare/original_unzipped/retrieved_2022_03_04/stanford_release_brainmask/mets_stanford_releaseMask_train",
                "label_root_dir":f"{paths['DATA']}/BrainMetShare/original_unzipped/retrieved_2022_03_04/stanford_release_brainmask/mets_stanford_releaseMask_train",
                "modality_names":["T1", "T1-spin-pre", "T1-spin-post", "T2-FLAIR"],
                "planes":[0],
                "clip_args":None,
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
        subj_dict, res_dict = proc.process_image_list(process_BrainMetShare_image,
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
        
        
global process_BrainMetShare_image
def process_BrainMetShare_image(item):
    try:
        dset_info = item['dset_info']
        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
        rtp = item["resolutions"] if item['redo_processed'] else put.check_proc_res(item)
        if len(rtp) > 0:
            image_dir = os.path.join(dset_info[item['subdset']]["image_root_dir"], item['image'])
            label_dirs = [os.path.join(image_dir, "seg", slice) for slice in os.listdir(os.path.join(image_dir, "seg"))]
            if item['load_images']:
                t1_im_dirs = [os.path.join(image_dir, "0", slice) for slice in os.listdir(os.path.join(image_dir, "0"))]
                t1_spin_pre_dirs = [os.path.join(image_dir, "1", slice) for slice in os.listdir(os.path.join(image_dir, "1"))]
                t1_spin_post_dirs = [os.path.join(image_dir, "2", slice) for slice in os.listdir(os.path.join(image_dir, "2"))]
                t2_flair_dirs = [os.path.join(image_dir, "3", slice) for slice in os.listdir(os.path.join(image_dir, "3"))]

                t1_images = np.stack([np.array(Image.open(im_dir).convert('L')) for im_dir in t1_im_dirs])
                t1_spin_pre_image = np.stack([np.array(Image.open(im_dir).convert('L')) for im_dir in t1_spin_pre_dirs])
                t1_spin_post_image = np.stack([np.array(Image.open(im_dir).convert('L')) for im_dir in t1_spin_post_dirs])
                t2_flair_image = np.stack([np.array(Image.open(im_dir).convert('L')) for im_dir in t2_flair_dirs])

                loaded_image = np.stack([t1_images, t1_spin_pre_image, t1_spin_post_image, t2_flair_image], -1)
                loaded_label = np.stack([np.array(Image.open(im_dir)) for im_dir in label_dirs])

                assert not (loaded_label is None), "Invalid Label"
                assert not (loaded_image is None), "Invalid Image"
            else:
                loaded_image = None
                loaded_label = np.stack([np.array(Image.open(im_dir)) for im_dir in label_dirs])

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