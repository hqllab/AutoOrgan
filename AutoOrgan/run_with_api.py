import os
import json
import torch
import inspect
import importlib
import multiprocessing

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from pathlib import Path
from joblib import Parallel, delayed
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from postprocess_gpu import work

def check_nnunet():
    nnunet_available = importlib.util.find_spec("nnunetv2") is not None
    if not nnunet_available :
        print('Please install nnunet,like pip install nnunetv2')

def keep_setting(source_image,origin,direction,spacing):
    source_image.SetOrigin(origin)
    source_image.SetDirection(direction)
    source_image.SetSpacing(spacing)
    return source_image

def supports_keyword_argument(func, keyword: str):
    signature = inspect.signature(func)
    parameters = signature.parameters
    return keyword in parameters

def predict(model_folder,file_path,device,folds=0,step_size=0.5,disable_tta=True,verbose=False,allow_tqdm=True):
    if supports_keyword_argument(nnUNetPredictor, "perform_everything_on_gpu"):
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_gpu=True,  # for nnunetv2<=2.2.1
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm
        )
    # nnUNet >= 2.2.2
    else:
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_device=False,  # for nnunetv2>=2.2.2
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm
        )
        
    predictor.initialize_from_trained_model_folder(model_folder,use_folds=[folds])        
    output_path = str(file_path).replace('resampled','segment')
    predictor.predict_from_files([[str(file_path)]], [str(output_path)],
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None,
                                 num_parts=1, part_id=0)
    return output_path
    
def resample(input_array,old_spacing,new_spacing,flag='image',cpus=-1):
    
    def _process_gradient(grad_idx):
        if flag == 'image':
            order = 3
        else:
            order = 0
        return ndimage.zoom(input_array[:, :, :, grad_idx], zoom, order=order)
    
    if type(new_spacing) is float:
        new_spacing = [new_spacing,] * 3   # for 3D and 4D
        new_spacing = np.array(new_spacing)
    
    if np.array_equal(old_spacing, new_spacing):
        return input_array

    zoom = old_spacing / new_spacing

    input_array = input_array[..., None]
    all_cpus_count = multiprocessing.cpu_count() if cpus == -1 else cpus
    result = Parallel(n_jobs=all_cpus_count)(delayed(_process_gradient)(grad_idx) for grad_idx in range(input_array.shape[3]))
    result = np.array(result).transpose(1, 2, 3, 0) 
    result = result[:,:,:,0]
    return result    

def run_segment(config_dict):
    
    device = config_dict.device
    input_folder = config_dict.input_folder
    output_folder = config_dict.output_folder
    
    # GPU check
    assert torch.cuda.is_available(), "No GPU to use."
    gpu_count = torch.cuda.device_count()
    gpu_device = torch.device(device)
    print(f"This computer have {gpu_count} GPUs,now use {gpu_device}")
    
    input_folder = Path(input_folder)
    file_list = [file for file in input_folder.iterdir() if (file.is_file() and (str(file).endswith("nii.gz") or str(file).endswith("nii")))]
    assert len(file_list) > 0,'No file in input folder.'

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    model_path = Path(config_dict.model_folder)
    assert model_path.exists(),'mode not exists.'
    
    temporary_folder = Path(config_dict.temporary_folder)
    
    json_path = model_path / 'nnUNetTrainer__nnUNetPlans__3d_fullres' / 'plans.json'
    data_json_path = model_path / 'nnUNetTrainer__nnUNetPlans__3d_fullres' / 'dataset.json'
    with open(json_path, 'r', encoding='utf-8') as file:
        json_dict = json.load(file)    
    
    with open(data_json_path, 'r', encoding='utf-8') as file:
        data_json_dict = json.load(file)        
    
    new_spacing = np.flip(np.array(json_dict['configurations']['3d_fullres']['spacing']))
    
    for file_path in file_list:
        file_image = sitk.ReadImage(file_path)
        file_array = sitk.GetArrayFromImage(file_image).astype(np.int32)
        
        old_spacing = np.array(file_image.GetSpacing())
        resampled_array = resample(file_array,old_spacing,new_spacing,flag='image',cpus=config_dict.use_cpus)
        
        resamplee_image = sitk.GetImageFromArray(resampled_array)
        resamplee_image = keep_setting(resamplee_image,file_image.GetOrigin(),file_image.GetDirection(),new_spacing)
        
        resampled_path = temporary_folder / ('resampled_' + file_path.name)
        sitk.WriteImage(resamplee_image,resampled_path)
        
        segment_file = predict(model_path / 'nnUNetTrainer__nnUNetPlans__3d_fullres',resampled_path,gpu_device)
        
        segment_image = sitk.ReadImage(segment_file)
        segment_array = sitk.GetArrayFromImage(segment_image).astype(np.uint8)
        
        resampled_segment_array = resample(segment_array,new_spacing,old_spacing,flag='mask',cpus=config_dict.use_cpus)
        post_array = work(resampled_segment_array,data_json_dict)
        post_array = post_array.astype(np.uint8)
        
        resampled_segment_image = sitk.GetImageFromArray(post_array)
        resampled_segment_image = keep_setting(resampled_segment_image,file_image.GetOrigin(),file_image.GetDirection(),new_spacing)
        resamplee_image.SetSpacing(old_spacing)
        sitk.WriteImage(resampled_segment_image,output_folder / file_path.name)

if __name__ == '__main__':
    
    config_dict = {}
    
    run_segment(config_dict)