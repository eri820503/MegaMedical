import submitit
from megamedical.utils.registry import paths

def run_submitit_job_array(datasets, func, timeout, mem, num_gpus=1):
    jobs = []
    executor = submitit.AutoExecutor(folder=f'{paths["ROOT"]}/bash/submitit')
    executor.update_parameters(timeout_min=timeout, mem_gb=mem, gpus_per_node=num_gpus, slurm_partition="sablab", slurm_wckey="")
    for dset in datasets:
        job = executor.submit(func, dset)
        jobs.append(job)
    return jobs

def return_empty_dict_copy(original_dict):
    new_dict = {}
    for key in original_dict.keys():
        new_dict[key] = original_dict[key]
        for sub_key in new_dict[key].keys():
            new_dict[key][sub_key] = 0
    return new_dict

def get_num_options(original_dict):
    num_options = 0
    for key in original_dict.keys():
        for sub_key in original_dict[key].keys():
            num_options = len(original_dict[key][sub_key])
    return num_options
