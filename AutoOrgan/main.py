import sys
sys.path.append('/home/wangnannan/workdir/AutoOrgan/')

import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from munch import DefaultMunch
from skimage.transform import resize

from AutoOrgan.preprocess import preprocess
from AutoOrgan.infer import OnnxInfer
from AutoOrgan.fixed import work

class Config():
    def __init__(self):
        self.input_path = '/home/wangnannan/nnunet_dir/nnUNet_raw/Dataset105_TotalBone/imagesTr'
        # self.input_path = '/home/wangnannan/nnunet_dir/nnUNet_raw/Dataset998_TotalRib/test_data'
        self.output_path = '/home/wangnannan/nnunet_dir/nnUNet_raw/Dataset998_TotalRib/my_code_infer'
        self.result_path = '/home/wangnannan/nnunet_dir/nnUNet_results/Dataset106_TotalBone/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
        # self.result_path = '/home/wangnannan/nnunet_dir/nnUNet_results/Dataset998_TotalRib/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres'
        self.use_gpu = True
        
        self.fold = '0'
        self.model_name = 'model.onnx'
        self.file_suffix = '*.nii.gz'
    
    def update(self, args):
        for k, v in vars(args).items():
            if hasattr(self, k):
                setattr(self, k, v)

class Predictor():
    def __init__(self,config_class:Config):
        self.config = config_class
        
        input_path = Path(self.config.input_path)
        if input_path.is_dir():
            self.infer_file_list = sorted(input_path.glob(self.config.file_suffix))[:]
        else:
            self.infer_file_list = [input_path]

        self.result_path = Path(self.config.result_path)
        with open(self.result_path / 'dataset.json') as f:
            self.dataset_json_data = DefaultMunch.fromDict(json.load(f))
        self.classes_count = len(self.dataset_json_data.labels)
        
        with open(self.result_path / 'plans.json') as f:
            self.plans_json_data = DefaultMunch.fromDict(json.load(f))
        
        self.model_path = self.result_path / f'fold_{self.config.fold}' /self.config.model_name
        self.output_path = Path(self.config.output_path)
        
        self.onnx_infer = OnnxInfer(self.classes_count,self.model_path,self.config.use_gpu)
        
    def run_infer(self):
        for index in range(len(self.infer_file_list)): 
            print(f'Run {index + 1} / {len(self.infer_file_list)} files, current file is {self.infer_file_list[index].name}')
            
            ct_image = sitk.ReadImage(self.infer_file_list[index])
            ct_array = sitk.GetArrayFromImage(ct_image)     
            spacing = ct_image.GetSpacing()
            
            ct_array,slicers,gaussian,ct_parameters_dict = preprocess(ct_array, spacing,self.plans_json_data)
            ct_array = self.onnx_infer.new_run_infer_ONNX(ct_array,slicers,gaussian)
            ct_array = self.postprocess(ct_array,ct_parameters_dict)
            # ct_array = work(ct_array,labels_dict)
            
            image = sitk.GetImageFromArray(ct_array)
            image.CopyInformation(ct_image)
            sitk.WriteImage(image,self.output_path / self.infer_file_list[index].name)

    def postprocess(self,image_data, parameters_dict):
        slicer_revert_padding = parameters_dict.pad_bbox
        crop_image_data = image_data[tuple([slice(None), *slicer_revert_padding])]
        crop_image_data = np.argmax(crop_image_data,0)
        
        crop_zoom_image_data = resize(crop_image_data,parameters_dict.shape_after_crop,order=0)
        
        slicer = tuple([slice(*i) for i in parameters_dict.crop_bbox])
        segmentation_reverted_cropping = np.zeros(parameters_dict.origin_shape,dtype=np.uint16)
        segmentation_reverted_cropping[slicer] = crop_zoom_image_data
        image_data = segmentation_reverted_cropping
        
        image_data = image_data.astype(np.uint16)
        return image_data

def main_entry():
    config_class = Config()
    my_infer = Predictor(config_class)
    my_infer.run_infer()

if __name__ == '__main__':
    main_entry()