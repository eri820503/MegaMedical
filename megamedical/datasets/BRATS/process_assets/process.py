import medpy.io
from tqdm.notebook import tqdm_notebook
import nibabel as nib
import numpy as np
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class BRATS:

    def __init__(self):
        self.name = "BRATS"
        self.dset_info = {
            "2016_LGG":{
                "main":"BRATS",
                "image_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/LGG",
                "label_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/LGG",
                "modality_names":["FLAIR","T1","T1c","T2"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "2016_HGG":{
                "main":"BRATS",
                "image_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/HGG",
                "label_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2016/BRATS2015_Training/HGG",
                "modality_names":["FLAIR","T1","T1c","T2"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "2021":{
                "main":"BRATS",
                "image_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021",
                "label_root_dir":f"{paths['DATA']}/BRATS/original_unzipped/2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021",
                "modality_names":["FLAIR","T1","T1c","T2"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
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
                        subj_folder = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        if dset_name == "2021":
                            if load_images:
                                flair_im_dir = os.path.join(subj_folder, f"{image}_flair.nii.gz")
                                t1_im_dir = os.path.join(subj_folder, f"{image}_t1.nii.gz")
                                t1c_im_dir = os.path.join(subj_folder, f"{image}_t1ce.nii.gz")
                                t2_im_dir = os.path.join(subj_folder, f"{image}_t2.nii.gz")
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_seg.nii.gz")

                                flair_image = nib.load(flair_im_dir).get_fdata()
                                t1_image = nib.load(t1_im_dir).get_fdata()
                                t1c_image = nib.load(t1c_im_dir).get_fdata()
                                t2_image = nib.load(t2_im_dir).get_fdata()

                                loaded_image = np.stack([flair_image, t1_image, t1c_image, t2_image], axis=-1)
                                loaded_label = nib.load(label_dir).get_fdata()
                                assert not (loaded_label is None), "Invalid Label"
                                assert not (loaded_image is None), "Invalid Image"
                            else:
                                label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_seg.nii.gz")
                                loaded_image = None
                                loaded_label = nib.load(label_dir).get_fdata()
                        else:
                            if load_images:
                                flair_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_Flair*/VSD.Brain.XX.O.MR_Flair*.mha"))[0]
                                t1_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T1*/VSD.Brain.XX.O.MR_T1*.mha"))[0]
                                t1c_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T1c*/VSD.Brain.XX.O.MR_T1c*.mha"))[0]
                                t2_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_T2*/VSD.Brain.XX.O.MR_T2*.mha"))[0]
                                label_dir = glob.glob(os.path.join(self.dset_info[dset_name]["label_root_dir"], image, "VSD.Brain_*/VSD.Brain_*.mha"))[0]

                                flair_image, _ = medpy.io.load(flair_im_dir)
                                t1_image, _ = medpy.io.load(t1_im_dir)
                                t1c_image, _ = medpy.io.load(t1c_im_dir)
                                t2_image, _ = medpy.io.load(t2_im_dir)

                                loaded_image = np.stack([flair_image, t1_image, t1c_image, t2_image], axis=-1)
                                loaded_label, _ = medpy.io.load(label_dir)
                                assert not (loaded_label is None), "Invalid Label"
                                assert not (loaded_image is None), "Invalid Image"
                            else:
                                label_dir = glob.glob(os.path.join(self.dset_info[dset_name]["label_root_dir"], image, "VSD.Brain_*/VSD.Brain_*.mha"))[0]

                                loaded_image = None
                                loaded_label, _ = medpy.io.load(label_dir)

                        proc_return = proc_func(proc_dir,
                                              version,
                                              dset_name,
                                              image, 
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
                    print(e)
                    #raise ValueError
            res_dict[resolution] = accumulator
        if accumulate:
            return proc_dir, res_dict