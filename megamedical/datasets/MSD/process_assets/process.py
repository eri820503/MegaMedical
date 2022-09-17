import nibabel as nib
from tqdm.notebook import tqdm_notebook
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class MSD:

    def __init__(self):
        self.name = "MSD"
        self.dset_info = {
            "BrainTumour":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/BrainTumour/retrieved_2018_07_03/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/BrainTumour/retrieved_2018_07_03/segs",
                "modality_names":["FLAIR", "T1w", "T1gd","T2w"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "Heart":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Heart/retrieved_2021_04_25/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Heart/retrieved_2021_04_25/segs",
                "modality_names":["Mono"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "Liver":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Liver/retrieved_2018_05_26/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Liver/retrieved_2018_05_26/segs",
                "modality_names":["PVP-CT"],
                "planes":[0, 1, 2],
                "clip_args":[-250,250],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
            "Hippocampus":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Hippocampus/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Hippocampus/retrieved_2021_04_22/segs",
                "modality_names":["Mono"],
                "planes":[0, 1, 2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "Prostate":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Prostate/retrieved_2018_05_31/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Prostate/retrieved_2018_05_31/segs",
                "modality_names":["T2","ADC"],
                "planes":[2],
                "clip_args": [0.5, 99.5],
                "norm_scheme":"MR",
                "do_clip":True,
                "proc_size":256
            },
            "Lung":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Lung/retrieved_2018_05_31/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Lung/retrieved_2018_05_31/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
            "Pancreas":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Pancreas/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Pancreas/retrieved_2021_04_22/segs",
                "modality_names":["PVP-CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
            "HepaticVessel":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/HepaticVessel/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/HepaticVessel/retrieved_2021_04_22/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
            "Spleen":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Spleen/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Spleen/retrieved_2021_04_22/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
            "Colon":{
                "main":"MSD",
                "image_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Colon/retrieved_2021_04_22/images",
                "label_root_dir":f"{paths['DATA']}/MSD/original_unzipped/Colon/retrieved_2021_04_22/segs",
                "modality_names":["CT"],
                "planes":[0, 1, 2],
                "clip_args":[-500,1000],
                "norm_scheme":"CT",
                "do_clip":True,
                "proc_size":256
            },
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
        proc_dir = os.path.join(paths['DATA'], self.name, "processed")
        image_list = os.listdir(self.dset_info[dset_name]["image_root_dir"])
        accumulator = []
        for image in tqdm_notebook(image_list, desc=f'Processing: {dset_name}'):
            try:
                proc_dir_template = os.path.join(proc_dir, f"midslice_v{version}", dset_name, "*", image)
                if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                    im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                    label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image)

                    assert os.path.isfile(im_dir), "Valid image dir required!"
                    assert os.path.isfile(label_dir), "Valid label dir required!"

                    if load_images:
                        loaded_image = nib.load(im_dir)
                        loaded_label = nib.load(label_dir)

                        if len(loaded_image.shape) == 3:
                            loaded_image = put.resample_nib(loaded_image)
                            loaded_label = put.resample_mask_to(loaded_label, loaded_image)

                        loaded_image = loaded_image.get_fdata()
                        loaded_label = loaded_label.get_fdata()
                        assert not (loaded_label is None), "Invalid Label"
                        assert not (loaded_image is None), "Invalid Image"
                    else:
                        loaded_image = None
                        loaded_label = nib.load(label_dir).get_fdata()

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