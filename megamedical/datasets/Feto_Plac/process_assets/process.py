from PIL import Image
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class Feto_Plac:

    def __init__(self):
        self.name = "Feto_Plac"
        self.dset_info = {
            "retreived_2022_03_09":{
                "main": "Feto_Plac",
                "image_root_dir":f"{paths['DATA']}/Feto_Plac/original_unzipped/retreived_2022_03_09/FetoscopyPlacentaDataset/Vessel_segmentation_annotations",
                "label_root_dir":f"{paths['DATA']}/Feto_Plac/original_unzipped/retreived_2022_03_09/FetoscopyPlacentaDataset/Vessel_segmentation_annotations",
                "modality_names":["Video"],
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
        video_list = sorted(os.listdir(self.dset_info[subdset]["image_root_dir"]))
        proc_dir = os.path.join(paths['ROOT'], "processed")
        res_dict = {}
        subj_dict = {}
        for resolution in resolutions:
            accumulator = []
            subj_accumulator = []
            for video in tqdm_notebook(video_list, desc=f'Processing: {subdset}'):
                vid_dir = os.path.join(self.dset_info[subdset]["image_root_dir"], video)
                frameset = sorted(os.listdir(os.path.join(vid_dir, "images")))
                for frame in frameset:
                    try:
                        # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                        template_root = os.path.join(proc_dir, f"res{resolution}", self.name)
                        mid_proc_dir_template = os.path.join(template_root, f"midslice_v{version}", subdset, "*/*", f"{video}_{frame}")
                        max_proc_dir_template = os.path.join(template_root, f"maxslice_v{version}", subdset, "*/*", f"{video}_{frame}")
                        if redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0):
                            im_dir = os.path.join(vid_dir, "images", frame)
                            label_dir = os.path.join(vid_dir, "masks_gt", f"{frame[:-4]}_mask.png")

                            if load_images:
                                loaded_image = np.array(Image.open(im_dir).convert('L'))
                                loaded_label = np.array(Image.open(label_dir).convert('L'))
                                assert not (loaded_label is None), "Invalid Label"
                                assert not (loaded_image is None), "Invalid Image"
                            else:
                                loaded_image = None
                                loaded_label = np.array(Image.open(label_dir).convert('L'))
                            
                            subj_name = f"{video}_{frame}"
                            proc_return = pps_function(proc_dir,
                                                        version,
                                                        subdset,
                                                        subj_name, 
                                                        loaded_image,
                                                        loaded_label,
                                                        self.dset_info[subdset],
                                                        show_hists=show_hists,
                                                        show_imgs=show_imgs,
                                                        res=resolution,
                                                        save=save)
                        
                            if accumulate:
                                accumulator.append(proc_return)
                                subj_accumulator.append(subj_name)
                    except Exception as e:
                        print(e)
                        #raise ValueError
            res_dict[resolution] = accumulator
            subj_dict[resolution] = subj_accumulator
        if accumulate:
            return proc_dir, subj_dict, res_dict