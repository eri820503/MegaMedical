import seaborn as sns
import os
from tqdm.notebook import tqdm_notebook
import universandbox as usd
import numpy as np

from megamedical.utils.registry import paths
import megamedical.utils as utils
from megamedical.utils.proc_utils import *

def vis_label_hists(datasets, slice_version):
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])

    sns.set(rc = {'figure.figsize':(30,5)})
    for dataset in datasets:
        try:
            fig_path = os.path.join(paths["DATA"], dataset, "processed/figures")
            if slice_version:
                subdsets = os.listdir(fig_path)
                for subdset in subdsets:
                    subdset_fig_dir = os.path.join(fig_path, subdset)
                    planes = list(set([pf.split("_")[-1] for pf in os.listdir(subdset_fig_dir)]))
                    for plane in planes:
                        max_dict_path = os.path.join(subdset_fig_dir, f"max_lab_dict_{plane}")
                        middle_dict_path = os.path.join(subdset_fig_dir, f"mid_lab_dict_{plane}")
                        
                        total_maxslice_label_dict = load_obj(max_dict_path)
                        total_midslice_label_dict = load_obj(middle_dict_path)
                        
                        plane_key = int(plane[0])
                        ax1 = sns.barplot(x=list(total_midslice_label_dict[plane_key].keys()), y=list(total_midslice_label_dict[plane_key].values()))
                        ax1.set(title=f"Midslice | Plane {plane_key} | Label Frequency for {dataset}/{subdset}.")
                        ax1.bar_label(ax1.containers[0])
                        plt.show()
                        ax2 = sns.barplot(x=list(total_maxslice_label_dict[plane_key].keys()), y=list(total_maxslice_label_dict[plane_key].values()))
                        ax2.set(title=f"Maxslice | Plane {plane_key} | Label Frequency for {dataset}/{subdset}.")
                        ax2.bar_label(ax2.containers[0])
                        plt.show()
            else:
                dict_path = os.path.join(fig_path, "lab_dict.pickle")
                frequency_label_dict = load_obj(dict_path)
                sns.set(rc = {'figure.figsize':(30,5)})
                ax = sns.barplot(x=list(frequency_label_dict.keys()), y=list(frequency_label_dict.values()))
                ax.bar_label(ax.containers[0])
                ax.set(title=f"Label Frequency for {dataset}.")
                plt.show()
        except Exception as e:
            print(e)
            continue

            
def vis_dataset(datasets,
                num_vis,
                size=3,
                minimum_label=0,
                version="4.0"):
    if datasets == "all":
        datasets = os.listdir(paths["DATA"])
        
    dataset_objects = [utils.build_dataset(ds) for ds in datasets]
    
    for d_idx, dataset in enumerate(datasets):
        try:
            proc_dataset_path = os.path.join(paths["DATA"], dataset, "processed")
            slice_path = os.path.join(proc_dataset_path, f"midslice_v{version}")
            for subdset in os.listdir(slice_path):
                modalities_dir = os.path.join(slice_path, subdset)
                for modality in os.listdir(modalities_dir):
                    subj_dir = os.path.join(modalities_dir, modality)
                    subjects = np.array(os.listdir(subj_dir))
                    for plane in dataset_objects[d_idx].dset_info[subdset]["planes"]:
                        midslice_info = np.load(os.path.join(proc_dataset_path, "label_info", subdset, f"midslice_pop_lab_amount_{plane}.npy"))
                        valid_subs = subjects[(midslice_info.sum(axis=1) > minimum_label)]
                        
                        # MIDSLICE VIS
                        if len(valid_subs) > num_vis:
                            print(f"Mid-slice Dataset: {dataset}, Subdset: {subdset}, Modality: {modality}, Plane: {plane}")
                            
                            chosen_subs = np.random.choice(valid_subs, num_vis)

                            img_vols = np.array([np.load(os.path.join(subj_dir, subj, f"img_128_{plane}.npy")) for subj in chosen_subs])
                            seg_vols = np.array([np.load(os.path.join(subj_dir, subj, f"seg_128_{plane}.npy")) for subj in chosen_subs])
                            
                            if img_vols.shape[-1] != 1:
                                midslice_segs = []
                                background = np.zeros((img_vols.shape[1], img_vols.shape[2], 1))
                                for msv in seg_vols:
                                    combined = np.argmax(np.concatenate([background, msv], axis=-1), axis=-1)
                                    midslice_segs.append(combined)
                            else:
                                midslice_segs = [msv.squeeze() for msv in midslice_seg_volumes]
                            midslice_segs = np.array(midslice_segs)
                            
                            midslices = np.concatenate([img_vols, midslice_segs])
                            
                            usd.utils.display_array(midslices, 
                                                    nrows=2, 
                                                    ncols=num_vis,
                                                    box_size=size,  
                                                    do_colorbars=True, 
                                                    cmap="gray")
                            plt.show()
                        
                        subj_dir = subj_dir.replace("midslice", "maxslice")
                        maxslice_info = np.load(os.path.join(proc_dataset_path, "label_info", subdset, f"maxslice_pop_lab_amount_{plane}.npy"))
                        # Currently have a bad artifact in some files that the first column was given an offset. Need reprocessing
                        offset = 1 if not maxslice_info[:,0].any() else 0
                        for lab_idx in range(maxslice_info.shape[1]):
                            lab_idx = lab_idx - offset
                            
                            valid_subs = subjects[(maxslice_info[:,lab_idx] > minimum_label)]
                
                            if len(valid_subs) > num_vis:
                                print(f"Max-slice Dataset: {dataset}, Subdset: {subdset}, Modality: {modality}, Plane: {plane}, Label: {lab_idx + offset}")
                                chosen_subs = np.random.choice(valid_subs, num_vis)
                                
                                img_vols= [np.load(os.path.join(subj_dir, subj, f"img_128_{plane}.npy")) for subj in chosen_subs]
                                seg_vols = [np.load(os.path.join(subj_dir, subj, f"seg_128_{plane}.npy")) for subj in chosen_subs]

                                maxslice_imgs = np.array([miv[..., lab_idx] for miv in img_vols])
                                maxslice_segs = np.array([msv[..., lab_idx] for msv in seg_vols])
                                
                                usd.utils.display_array(np.concatenate([maxslice_imgs, maxslice_segs]),  
                                                        nrows=2, 
                                                        ncols=num_vis,
                                                        box_size=size,  
                                                        do_colorbars=True, 
                                                        cmap="gray")
                                plt.show()
        except Exception as e:
            print(e)
            #raise ValueError
            continue
                    

            

        
        

