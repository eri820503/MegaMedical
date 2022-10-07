from skimage import color
import scipy.io
import cv2
from tqdm.notebook import tqdm_notebook
import matplotlib.pyplot as plt
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put

class HMC_QU:

    def __init__(self):
        self.name = "HMC_QU"
        self.dset_info = {
            "retrieved_09_29_2022":{
                "main":"HMC_QU",
                "image_root_dir": f"{paths['DATA']}/HMC_QU/original_unzipped/retrieved_09_29_2022/HMC-QU/A4C",
                "label_root_dir": f"{paths['DATA']}/HMC_QU/original_unzipped/retrieved_09_29_2022/LV_Masks",
                "modality_names": ["Ultrasound"],
                "planes": [0],
                "clip_args": None,
                "norm_scheme": None
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
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        proc_dir = os.path.join(paths['ROOT'], "processed")
        res_dict = {}
        for resolution in resolutions:
            accumulator = []
            for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
                try:
                    # template follows processed/resolution/dset/midslice/subset/modality/plane/subject
                    template_root = os.path.join(proc_dir, f"res{resolution}", self.name)
                    mid_proc_dir_template = os.path.join(template_root, f"midslice_v{version}", dset_name, "*/*", image)
                    max_proc_dir_template = os.path.join(template_root, f"maxslice_v{version}", dset_name, "*/*", image)
                    if redo_processed or (len(glob.glob(mid_proc_dir_template)) == 0) or (len(glob.glob(max_proc_dir_template)) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], f"Mask_{image}".replace("avi", "mat"))
                        
                        loaded_labeled_video = scipy.io.loadmat(label_dir)["predicted"]
                        if load_images:
                            cap = cv2.VideoCapture(im_dir)
                        for frame in range(loaded_labeled_video.shape[0]):
                            loaded_label = loaded_labeled_video[frame, ...]
                            if load_images:
                                #Note that the images are different sizes than the labels, this is ok
                                #and handeled in processing.
                                _, uncropped_loaded_image = cap.read(frame)
                                half_x_dist = (uncropped_loaded_image.shape[1] - uncropped_loaded_image.shape[0])//2
                                loaded_image = color.rgb2gray(uncropped_loaded_image)[:,half_x_dist:-half_x_dist]
                            else:
                                loaded_image = None

                            proc_return = proc_func(proc_dir,
                                                      version,
                                                      dset_name,
                                                      f"{image}_frame{frame}", 
                                                      loaded_image,
                                                      loaded_label,
                                                      self.dset_info[dset_name],
                                                      show_hists=show_hists,
                                                      show_imgs=show_imgs,
                                                      res=resolution,
                                                      save=save)

                            if accumulate:
                                accumulator.append(proc_return)
                except Exception as e:
                    continue
                    #print(e)
                    #raise ValueError
            res_dict[resolution] = accumulator
        if accumulate:
            return proc_dir, res_dict
