from PIL import Image
from tqdm import tqdm
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
                "image_root_dir":f"{paths['DATA']}/Feto_Plac/processed/original_unzipped/retreived_2022_03_09/FetoscopyPlacentaDataset/Vessel_segmentation_annotations",
                "label_root_dir":f"{paths['DATA']}/Feto_Plac/processed/original_unzipped/retreived_2022_03_09/FetoscopyPlacentaDataset/Vessel_segmentation_annotations",
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
            for video in image_list:
                vid_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], video)
                for frame in os.listdir(os.path.join(vid_dir, "images")):
                    image = f"{video}_{frame}"
                    try:
                        if redo_processed or (len(glob.glob(os.path.join(processed_dir, "*", image))) == 0):
                            im_dir = os.path.join(vid_dir, "images", frame)
                            label_dir = os.path.join(vid_dir, "masks_gt", f"{frame[:-4]}_mask.png")

                            loaded_image = np.array(Image.open(im_dir).convert('L'))
                            loaded_label = np.array(Image.open(label_dir).convert('L'))

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