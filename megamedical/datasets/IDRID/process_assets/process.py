from PIL import Image
from tqdm import tqdm
import glob
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


class IDRID:

    def __init__(self):
        self.name = "IDRID"
        self.dset_info = {
            "retreived_2022_03_04":{
                "main": "IDRID",
                "image_root_dir":f"{paths['DATA']}/IDRID/processed/original_unzipped/retreived_2022_03_04/A. Segmentation/1. Original Images/a. Training Set",
                "label_root_dir":f"{paths['DATA']}/IDRID/processed/original_unzipped/retreived_2022_03_04/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set",
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
                        ma_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], "1. Microaneurysms", f"{image[:-4]}_MA.tif")
                        he_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], "2. Haemorrhages", f"{image[:-4]}_HE.tif")
                        ex_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], "3. Hard Exudates", f"{image[:-4]}_EX.tif")
                        se_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], "4. Soft Exudates", f"{image[:-4]}_SE.tif")
                        od_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], "5. Optic Disc", f"{image[:-4]}_OD.tif")

                        loaded_image = np.array(Image.open(im_dir).convert('L'))

                        loaded_label = np.array(Image.open(ma_dir))
                        if os.path.exists(he_dir):
                            he = np.array(Image.open(he_dir))*2
                            loaded_label += he
                        if os.path.exists(ex_dir):
                            ex = np.array(Image.open(ex_dir))*3
                            loaded_label += ex
                        if os.path.exists(se_dir):
                            se = np.array(Image.open(se_dir))*4
                            loaded_label += se
                        if os.path.exists(od_dir):
                            od = np.array(Image.open(od_dir))*5
                            loaded_label += od

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