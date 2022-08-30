import imageio as io
import nrrd
from tqdm.notebook import tqdm_notebook
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class SCD:

    def __init__(self):
        self.name = "SCD"
        self.dset_info = {
            "LAS":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAS/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAS/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "labels": [1,2,3],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "LAF_Pre":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Pre/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Pre/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "labels": [1,2,3],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "LAF_Post":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Post/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/LAF_Post/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "labels": [1,2,3],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "VIS_pig":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_pig/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_pig/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "labels": [1,2,3],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "VIS_human":{
                "main":"SCD",
                "image_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_human/retrieved_2021_11_08",
                "label_root_dir":f"{paths['DATA']}/SCD/original_unzipped/VIS_human/retrieved_2021_11_08",
                "modality_names":["MRI"],
                "planes":[2],
                "labels": [1,2,3],
                "clip_args":None,
                "norm_scheme":"MR",
                "do_clip":True,
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
                  redo_processed=True):
        assert not(version is None and save_slices), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(self.name, dset_name, save_slices, version, self.dset_info[dset_name])
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            for image in image_list:
                try:
                    proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        if dset_name=="LAS":
                            im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"],image) + "/image.mhd"
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + "/gt_binary.mhd"
                            loaded_image = io.imread(im_dir, plugin='simpleitk')
                            loaded_label = io.imread(label_dir, plugin='simpleitk')
                        else:
                            if dset_name=="LAF_Pre":
                                im_dir = os.path.join(self.dset_info["image_root_dir"],image) + f"/de_a_{image[1:]}.nrrd"
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + f"/la_seg_a_{image[1:]}.nrrd"
                            elif dset_name=="LAF_Post":
                                im_dir = os.path.join(self.dset_info["image_root_dir"],image) + f"/de_b_{image[1:]}.nrrd"
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + f"/la_seg_b_{image[1:]}.nrrd"
                            elif dset_name in ["VIS_pig","VIS_human"]:
                                im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"],image) + "/" + image + "_de.nrrd"
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"],image) + "/" + image + "_myo.nrrd"
                            loaded_image, _ = nrrd.read(im_dir)
                            loaded_label, _ = nrrd.read(label_dir)

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        proc_return = proc_func(proc_dir,
                                              version,
                                              dset_name,
                                              image, 
                                              loaded_image,
                                              loaded_label,
                                              self.dset_info[dset_name],
                                              show_hists=show_hists,
                                              show_imgs=show_imgs,
                                              save=save)

                    if accumulate:
                        accumulator.append(proc_return)
            except Exception as e:
                print(e)
                #raise ValueError
        if accumulate:
            return proc_dir, accumulator