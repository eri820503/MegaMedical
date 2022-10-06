import nibabel as nib
from tqdm.notebook import tqdm_notebook
import numpy as np
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class ISLES:

    def __init__(self):
        self.name = "ISLES"
        self.dset_info = {
            "ISLES2017":{
                "main":"ISLES",
                "image_root_dir":f"{paths['DATA']}/ISLES/original_unzipped/ISLES2017/training",
                "label_root_dir":f"{paths['DATA']}/ISLES/original_unzipped/ISLES2017/training",
                "modality_names":["ADC","MIT","TTP","Tmax","rCBF","rCBV"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR"
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

                        ADC_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_ADC*/VSD.Brain.XX.O.MR_ADC*.nii"))[0]
                        MIT_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_MTT*/VSD.Brain.XX.O.MR_MTT*.nii"))[0]
                        TTP_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_TTP*/VSD.Brain.XX.O.MR_TTP*.nii"))[0]
                        Tmax_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_Tmax*/VSD.Brain.XX.O.MR_Tmax*.nii"))[0]
                        rCBF_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_rCBF*/VSD.Brain.XX.O.MR_rCBF*.nii"))[0]
                        rCBV_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_rCBV*/VSD.Brain.XX.O.MR_rCBV*.nii"))[0]

                        label_dir = glob.glob(os.path.join(self.dset_info[dset_name]["label_root_dir"], image, "VSD.Brain.XX.O.OT*/VSD.Brain.XX.O.OT*.nii"))[0]

                        ADC = put.resample_nib(nib.load(ADC_im_dir))
                        MIT = put.resample_nib(nib.load(MIT_im_dir))
                        TTP = put.resample_nib(nib.load(TTP_im_dir))
                        Tmax = put.resample_nib(nib.load(Tmax_im_dir))
                        rCBF = put.resample_nib(nib.load(rCBF_im_dir))
                        rCBV = put.resample_nib(nib.load(rCBV_im_dir))

                        loaded_label = put.resample_mask_to(nib.load(label_dir), ADC)

                        ADC = ADC.get_fdata()
                        MIT = MIT.get_fdata()
                        TTP = TTP.get_fdata()
                        Tmax = Tmax.get_fdata()
                        rCBF = rCBF.get_fdata()
                        rCBV = rCBV.get_fdata()

                        loaded_image = np.stack([ADC, MIT, TTP, Tmax, rCBF, rCBV], -1)
                        loaded_label = loaded_label.get_fdata()

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