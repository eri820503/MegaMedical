#Concurrency
from concurrent.futures import ProcessPoolExecutor
#Progress vis
from tqdm.notebook import tqdm_notebook


def process_image_list(process_DATASET_image,
                       proc_dir,
                       image_list,
                       parallelize,
                       pps_function,
                       resolutions,
                       dataset,
                       subdset,
                       dset_info,
                       redo_processed,
                       load_images,
                       show_hists,
                       version,
                       show_imgs,
                       accumulate,
                       save): 
    
    item_list = [{
                  "image": image,
                  "pps_function": pps_function,
                  "resolutions": resolutions,
                  "proc_dir": proc_dir,
                  "dataset": dataset,
                  "subdset": subdset,
                  "dset_info": dset_info,
                  "redo_processed": redo_processed,
                  "load_images": load_images,
                  "show_hists": show_hists,
                  "version": version,
                  "show_imgs": show_imgs,
                  "save": save
                  } for image in image_list]
    
    res_dict = {}
    subj_accumulator = []
    if parallelize:
        with ProcessPoolExecutor(max_workers=16) as executor:
            for (proc_return, subj_name) in executor.map(process_DATASET_image, item_list):
                if accumulate and subj_name is not None:
                    for res in resolutions:
                        res_dict[res].append(proc_return[res])
                    subj_accumulator.append(subj_name)
    else:
        for item in tqdm_notebook(item_list, desc=f'Processing: {subdset}'):
            proc_return, subj_name = process_DATASET_image(item)
            if accumulate and subj_name is not None:
                for res in resolutions:
                    res_dict[res].append(proc_return[res])
                subj_accumulator.append(subj_name)  
    
    return subj_dict, res_dict