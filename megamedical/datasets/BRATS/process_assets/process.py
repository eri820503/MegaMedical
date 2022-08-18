import medpy.io
from tqdm import tqdm
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
                  version=None,
                  show_hists=False,
                  show_imgs=False,
                  save_slices=False,
                  redo_processed=True):
        assert not(version is None and save_slices), "Must specify version for saving."
        assert dset_name in self.dset_info.keys(), "Sub-dataset must be in info dictionary."
        proc_dir = pps.make_processed_dir(self.name, dset_name, save_slices, version, self.dset_info[dset_name])
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        subj_folder = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        if dset_name == "2021":
                            flair_im_dir = os.path.join(subj_folder, f"{image}_flair.nii.gz")
                            t1_im_dir = os.path.join(subj_folder, f"{image}_t1.nii.gz")
                            t1c_im_dir = os.path.join(subj_folder, f"{image}_t1ce.nii.gz")
                            t2_im_dir = os.path.join(subj_folder, f"{image}_t2.nii.gz")

                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image, f"{image}_seg.nii.gz")

                            flair_image = put.resample_nib(nib.load(flair_im_dir))
                            t1_image = put.resample_nib(nib.load(t1_im_dir))
                            t1c_image = put.resample_nib(nib.load(t1c_im_dir))
                            t2_image = put.resample_nib(nib.load(t2_im_dir))
                            loaded_label = put.resample_mask_to(nib.load(label_dir), flair_image)
                            
                            flair_image = flair_image.get_fdata()
                            t1_image = t1_image.get_fdata()
                            t1c_image = t1c_image.get_fdata()
                            t2_image = t2_image.get_fdata()
                            loaded_label = loaded_label.get_fdata()

                            loaded_image = np.stack([flair_image, t1_image, t1c_image, t2_image], axis=-1)
                        else:
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

                        assert not (loaded_image is None), "Invalid Image"
                        assert not (loaded_label is None), "Invalid Label"

                        pps.produce_slices(proc_dir,
                                          version,
                                          dset_name,
                                          image, 
                                          loaded_image,
                                          loaded_label,
                                          self.dset_info[dset_name],
                                          show_hists=show_hists,
                                          show_imgs=show_imgs,
                                          save_slices=save_slices)
                except Exception as e:
                    print(e)
                    raise ValueError
                pbar.update(1)
        pbar.close()