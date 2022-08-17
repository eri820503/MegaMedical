from PIL import Image
from tqdm import tqdm
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class ROSE:

    def __init__(self):
        self.name = "ROSE"
        self.dset_info = {
            "ROSE-1-DVC":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-1-DVC/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-1-DVC/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
            "ROSE-1-SVC":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-1-SVC/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-1-SVC/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
            "ROSE-1-SVC_DVC":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-1-SVC_DVC/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-1-SVC_DVC/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
                "proc_size":256
            },
            "ROSE-2":{
                "main": "ROSE",
                "image_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-2/img",
                "label_root_dir":f"{paths['DATA']}/ROSE/processed/original_unzipped/ROSE-2/gt",
                "modality_names":["Retinal"],
                "planes":[0],
                "clip_args":None,
                "norm_scheme":None,
                "do_clip":False,
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
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image)
                        if dset_name in ["ROSE-1-DVC", "ROSE-1-SVC", "ROSE-2"]:
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image)
                        else:
                            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], image.replace(".png",".tif"))

                        loaded_image = np.array(Image.open(im_dir).convert('L'))
                        loaded_label = np.array(Image.open(label_dir))

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