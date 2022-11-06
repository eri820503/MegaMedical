import neurite_sandbox as nes
from tqdm.notebook import tqdm_notebook
import neurite as ne
import numpy as np
import tensorflow as tf
import os


def perlin_generation(num_to_gen,
                      num_labels_range=[5,20],
                      max_blur_std=30,
                      shapes_im_max_std_range=[0.5, 5],
                      shapes_warp_max_std_range=[4.0, 15.0],
                      std_min_range=[0.01, 0.1],
                      std_max_range=[0.2, 1],
                      lab_int_interimage_std_range=[0.01, 0.1],
                      warp_std_range=[1, 20],
                      bias_res_range=[32, 50],
                      bias_std_range=[0.1, 1.0],
                      blur_std_range=[0.5, 5],
                      min_density=0.001,
                      ):

    # Gen parameters
    num_labels = np.random.randint(low=num_labels_range[0], high=num_labels_range[1])
    shapes_im_max_std = np.random.uniform(shapes_im_max_std_range[0], shapes_im_max_std_range[1])
    shapes_warp_max_std = np.random.uniform(shapes_warp_max_std_range[0], shapes_warp_max_std_range[1])
    std_min = np.random.uniform(std_min_range[0], std_min_range[1])
    std_max = np.random.uniform(std_max_range[0], std_max_range[1])
    lab_int_interimage_std = np.random.uniform(lab_int_interimage_std_range[0], lab_int_interimage_std_range[1])
    warp_std = np.random.uniform(warp_std_range[0], warp_std_range[1])
    bias_res = np.random.uniform(bias_res_range[0], bias_res_range[1])
    bias_std = np.random.uniform(bias_std_range[0], bias_std_range[1])
    blur_std = np.random.uniform(blur_std_range[0], blur_std_range[1])

    # Gen tasks
    images, label_maps, lab = nes.tf.utils.synth.perlin_nshot_task(in_shape=(256,256),
                                                                  num_gen=num_to_gen,
                                                                  num_label=num_labels,
                                                                  shapes_im_scales=(32, 64, 128),
                                                                  shapes_warp_scales=(16, 32, 64, 128),
                                                                  shapes_im_max_std=shapes_im_max_std,
                                                                  shapes_warp_max_std=shapes_warp_max_std,
                                                                  min_int=0,
                                                                  max_int=1,
                                                                  std_min=std_min,
                                                                  std_max=std_max,
                                                                  lab_int_interimage_std=lab_int_interimage_std,
                                                                  warp_std=warp_std,
                                                                  warp_res=(8, 16, 32, 64),
                                                                  bias_res=bias_res,
                                                                  bias_std=bias_std,
                                                                  blur_std=blur_std)


    flattened_labels = [np.argmax(f, -1) for f in label_maps]
    all_labels = np.unique(lab).tolist()

    background_labels = []
    for lab in all_labels:
        for lm in label_maps:
            if np.mean(lm[..., lab]) < min_density:
                all_labels.remove(lab)
                background_labels.append(background_labels)
                break

    assert len(all_labels) > 0

    label_map_big_labels = [f[..., np.array(all_labels)] for f in label_maps]

    label_map_big_labels_wbg = []
    foreground_labels = []
    background_labels = []
    for f in label_map_big_labels:
        foreground_label = (np.sum(f, axis=-1))[..., np.newaxis]
        background_label = (1-np.sum(f, axis=-1))[..., np.newaxis]
        foreground_labels.append(foreground_label)
        background_labels.append(background_label)
        new_f = np.concatenate([background_label, f], -1)
        label_map_big_labels_wbg.append(new_f)
    flattened_label_map_big_labels_wbg = [np.argmax(f, -1) for f in label_map_big_labels_wbg]

    return images, flattened_label_map_big_labels_wbg


def synth_subjects(
    num_tasks: int,
    num_subjects: int = 100,
    min_density: float = 0.001
):
    root = "/storage/megamedical/v4-raw/Synthetic1000/original_unzipped"
    for task in tqdm_notebook(range(num_tasks)):
        task_root = os.path.join(root, f"task_{task}")
        images, label_maps = perlin_generation(num_to_gen=num_subjects,
                                               min_density=min_density)

        for subj_idx, (image, label_map) in enumerate(zip(images, label_maps)):
            img_root = os.path.join(task_root, "imgs")
            label_root = os.path.join(task_root, "segs")
            if not os.path.exists(img_root):
                os.makedirs(img_root)
            if not os.path.exists(label_root):
                os.makedirs(label_root)
            np.save(os.path.join(img_root, f"subj_{subj_idx}"), image)
            np.save(os.path.join(label_root, f"subj_{subj_idx}"), label_map)


if __name__ == "__main__":
    import typer

    typer.run(synth_subjects)
