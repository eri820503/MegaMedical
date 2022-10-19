#Concurrency
from concurrent.futures import ProcessPoolExecutor
#Progress vis
from tqdm.notebook import tqdm_notebook


def process_image_list(process_DATASET_image,
                       proc_dir,
                       task,
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
                       save,
                       preloaded_images=None,
                       preloaded_labels=None): 
    
    item_list = [{
                  "image": image,
                  "task": task, 
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
                  "save": save,
                  "image_array": preloaded_images,
                  "label_array": preloaded_labels
                  } for image in image_list]
    
    res_dict = {res:[] for res in resolutions}
    subj_dict = {res:[] for res in resolutions}
    
    if parallelize:
        with ProcessPoolExecutor(max_workers=16) as executor:
            for (proc_return, subj_name) in tqdm_notebook(executor.map(process_DATASET_image, item_list), 
                                                          total=len(item_list),
                                                          desc=f'Processing: {subdset}'):
                if accumulate and subj_name is not None:
                    for res in resolutions:
                        res_dict[res].append(proc_return[res])
                        subj_dict[res].append(subj_name)
    else:
        for item in tqdm_notebook(item_list, desc=f'Processing: {subdset}'):
            proc_return, subj_name = process_DATASET_image(item)
            if accumulate and subj_name is not None:
                for res in resolutions:
                    res_dict[res].append(proc_return[res])
                    subj_dict[res].append(subj_name)  
    
    return subj_dict, res_dict