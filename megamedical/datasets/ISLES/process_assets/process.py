import nibabel as nib
from tqdm import tqdm
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
                "image_root_dir":f"{paths['DATA']}/ISLES/processed/original_unzipped/ISLES2017/training",
                "label_root_dir":f"{paths['DATA']}/ISLES/processed/original_unzipped/ISLES2017/training",
                "modality_names":["ADC","MIT","TTP","Tmax","rCBF","rCBV"],
                "planes":[2],
                "clip_args":None,
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
        proc_dir = pps.make_processed_dir(self.name, dset_name, save_slices, version)
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        with tqdm(total=len(image_list), desc=f'Processing: {dset_name}', unit='image') as pbar:
            for image in image_list:
                try:
                    if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                        subj_folder = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)

                        ADC_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_ADC*/VSD.Brain.XX.O.MR_ADC*.nii"))[0]
                        MIT_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_MTT*/VSD.Brain.XX.O.MR_MTT*.nii"))[0]
                        TTP_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_TTP*/VSD.Brain.XX.O.MR_TTP*.nii"))[0]
                        Tmax_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_Tmax*/VSD.Brain.XX.O.MR_Tmax*.nii"))[0]
                        rCBF_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_rCBF*/VSD.Brain.XX.O.MR_rCBF*.nii"))[0]
                        rCBV_im_dir = glob.glob(os.path.join(subj_folder, "VSD.Brain.XX.O.MR_rCBV*/VSD.Brain.XX.O.MR_rCBV*.nii"))[0]

                        label_dir = glob.glob(os.path.join(self.dset_info[dset_name]["label_root_dir"], image, "VSD.Brain.XX.O.OT*/VSD.Brain.XX.O.OT*.nii"))[0]

                        ADC = np.array(nib.load(ADC_im_dir).dataobj)
                        MIT = np.array(nib.load(MIT_im_dir).dataobj)
                        TTP = np.array(nib.load(TTP_im_dir).dataobj)
                        Tmax = np.array(nib.load(Tmax_im_dir).dataobj)
                        rCBF = np.array(nib.load(rCBF_im_dir).dataobj)
                        rCBV = np.array(nib.load(rCBV_im_dir).dataobj)

                        loaded_image = np.stack([ADC, MIT, TTP, Tmax, rCBF, rCBV], -1)
                        loaded_label = np.array(nib.load(label_dir).dataobj)

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
                                          show_imgs=show_imgs)
                except Exception as e:
                    print(e)
                pbar.update(1)
        pbar.close()