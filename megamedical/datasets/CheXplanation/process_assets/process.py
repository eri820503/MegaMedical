import nibabel as nib
from tqdm import tqdm
import glob
from PIL import Image
import numpy as np
import json
import os

#New line!
from megamedical.src import preprocess_scripts as pps
from megamedical.utils.registry import paths
from megamedical.utils import proc_utils as put


def create_mask(polygons, img_dims):
    """
    Creates a binary mask (of the original matrix size) given a list of polygon
    annotations format.
    Args:
        polygons (list): [[[x11,y11],[x12,y12],...[x1n,y1n]],...]
    Returns:
        mask (np.array): binary mask, 1 where the pixel is predicted to be the,
    pathology, 0 otherwise
    """
    poly = Image.new('1', (img_dims[1], img_dims[0]))
    print(polygons)
    for polygon in polygons:
        coords = [(point[0], point[1]) for point in polygon]
        ImageDraw.Draw(poly).polygon(coords,  outline=1, fill=1)

    binary_mask = np.array(poly, dtype="int")
    return binary_mask


class CheXplanation:

    def __init__(self):
        self.name = "Chexplanation"
        self.dset_info = {
            "retreived_2022_03_04":{
                "main":"CoNSeP",
                "image_root_dir":f"{paths['DATA']}/CheXplanation/original_unzipped/v1.0/CheXplanation/CheXpert-v1.0/valid",
                "label_root_dir":f"{paths['DATA']}/CheXplanation/original_unzipped/v1.0/CheXplanation",
                "modality_names":["NA"],
                "planes":[0],
                "clip_args": None,
                "norm_scheme": None,
                "do_clip": False,
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
            label_dir = os.path.join(self.dset_info[dset_name]["label_root_dir"], "gt_segmentation_val.json")
            label_file = open(label_dir)
            label_json = json.load(label_file)
            for image in image_list:
                try:
                    proc_dir_template = os.path.join(proc_dir, f"megamedical_v{version}", dset_name, "*", image)
                    if redo_processed or (len(glob.glob(proc_dir_template)) == 0):
                        im_dir = os.path.join(self.dset_info[dset_name]["image_root_dir"], image, "study1/view1_frontal.jpg")
                        
                        assert os.path.isfile(im_dir), "Valid image dir required!"
                        
                        loaded_image = np.array(Image.open(im_dir).convert('L'))
                        subj_dict = label_json[f"{image}_study1_view1_frontal"]
                        for label in subj_dict.keys():
                            loaded_bin_mask = create_mask(subj_dict[label]["counts"], subj_dict[label]["size"])

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