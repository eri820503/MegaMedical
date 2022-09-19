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
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            }
        }

    def proc_func(self,
                  dset_name,
                  proc_func,
                  load_images=True,
                  accumulate=False,
                  version=None,
                  show_imgs=False,
                  save=False,
                  show_hists=False,
                  resolutions=None,
                  redo_processed=True):
        assert not(version is None and save), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = os.path.join(paths['ROOT'], "processed")
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        accumulator = []
        for video in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            vid_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], video)
            for frame in os.listdir(os.path.join(vid_dir, "images")):
                image = f"{video}_{frame}"
                try:
                    proc_dir_template = os.path.join(proc_dir, f"midslice_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
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

                        proc_return = proc_func(proc_dir,
                                          version,
                                          dset_name,
                                          image, 
                                          loaded_image,
                                          loaded_label,
                                          self.dset_info[dset_name],
                                          show_hists=show_hists,
                                          show_imgs=show_imgs,
                                          resolutions=resolutions,
                                          save=save)

                        if accumulate:
                            accumulator.append(proc_return)
                except Exception as e:
                    print(e)
                    #raise ValueError
        if accumulate:
            return proc_dir, accumulator