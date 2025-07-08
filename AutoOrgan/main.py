import sys
sys.path.append('/home/wangnannan/workdir/AutoOrgan/')

import argparse
import json
import queue
import threading
import SimpleITK as sitk
from pathlib import Path
from munch import DefaultMunch

from infer import OnnxInfer
from preprocess import preprocess

class Predictor():
    def __init__(self,config_class):
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
        
        self.model_path = self.result_path / self.config.model_name
        # self.model_path = self.result_path / f'fold_{self.config.fold}' /self.config.model_name
        self.output_path = Path(self.config.output_path)
        
        self.onnx_infer = OnnxInfer(self.classes_count,self.model_path,self.config.use_gpu)
        
    def load_and_preprocess(self,image_path):
        ct_image = sitk.ReadImage(image_path)
        ct_array = sitk.GetArrayFromImage(ct_image)     
        spacing = ct_image.GetSpacing()
        
        ct_array,slicers,gaussian,parameters_dict = preprocess(ct_array, spacing,self.plans_json_data)
        return (ct_array,slicers,gaussian,parameters_dict,ct_image,image_path.name)

    def load_queue(self,file_paths,data_queue):
        for path in file_paths:
            data = self.load_and_preprocess(path)
            data_queue.put(data)
        data_queue.put(None)     
        
    # def run_infer(self):
    #     data_queue = queue.Queue(maxsize=2) 
    #     loader = threading.Thread(target=self.load_queue, args=(self.infer_file_list, data_queue))
    #     loader.start()
        
    #     while True:
    #         data = data_queue.get()
    #         if data is None:
    #             break
    #         print(f'Run current file is {data[5]}')
    #         ct_array = self.onnx_infer.run_infer_ONNX(data[0],data[1],data[2],data[3])
    #         image = sitk.GetImageFromArray(ct_array)
    #         image.CopyInformation(data[4])
    #         sitk.WriteImage(image,self.output_path / data[5])
    
    def run_infer(self):
        for index in range(len(self.infer_file_list)): 
            print(f'Run {index + 1} / {len(self.infer_file_list)} files, current file is {self.infer_file_list[index].name}')
            
            ct_image = sitk.ReadImage(self.infer_file_list[index])
            ct_array = sitk.GetArrayFromImage(ct_image)     
            spacing = ct_image.GetSpacing()
            
            ct_array,slicers,gaussian,parameters_dict = preprocess(ct_array, spacing,self.plans_json_data)
            ct_array = self.onnx_infer.run_infer_ONNX(ct_array,slicers,gaussian,parameters_dict)
            
            image = sitk.GetImageFromArray(ct_array)
            image.CopyInformation(ct_image)
            sitk.WriteImage(image,self.output_path / self.infer_file_list[index].name)
        
def main_entry():
    parser = argparse.ArgumentParser(description="这是一个示例程序")

    # parser.add_argument("-i",'--input_path', type=str,default='/home/wangnannan/nnunet_dir/nnUNet_raw/Dataset105_TotalBone/imagesTr', help="The input directory can be a single file or a folder")
    # parser.add_argument('-o',"--output_path", type=str, required=False,default='/home/wangnannan/nnunet_dir/nnUNet_raw/Dataset998_TotalRib/my_code_infer', help="Output path")
    # parser.add_argument('-r',"--result_path", type=str, required=False,default='/home/wangnannan/nnunet_dir/nnUNet_results/Dataset106_TotalBone/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres', help="The path of the nnUnet result path")
    # parser.add_argument('-f',"--fold", type=str,default='0', help="Fold number")    
    # parser.add_argument('-m',"--model_name", type=str, default='model.onnx',help="The name of the model used")    
    # parser.add_argument('-fs',"--file_suffix", type=str,default="*.nii.gz", help="You want the suffix of the inference files")    
    # parser.add_argument('-g',"--use_gpu", action="store_true",default=True, help="Whether to use a GPU for acceleration")    

    # parser.add_argument("-i",'--input_path', type=str,default=r'C:\Users\27321\Desktop\AutoOrgan\input', help="The input directory can be a single file or a folder")
    # parser.add_argument('-o',"--output_path", type=str, required=False,default=r'C:\Users\27321\Desktop\AutoOrgan\output', help="Output path")
    # parser.add_argument('-r',"--result_path", type=str, required=False,default=r'C:\Users\27321\Desktop\AutoOrgan\result', help="The path of the model path")
    # parser.add_argument('-m',"--model_name", type=str, default='model.onnx',help="The name of the model used")    
    # parser.add_argument('-fs',"--file_suffix", type=str,default="*.nii.gz", help="You want the suffix of the inference files")    
    # parser.add_argument('-g',"--use_gpu", action="store_true",default=True, help="Whether to use a GPU for acceleration")
    
    parser.add_argument("-i",'--input_path', type=str,required=True, help="The input directory can be a single file or a folder")
    parser.add_argument('-o',"--output_path", type=str, required=True, help="Output path")
    parser.add_argument('-r',"--result_path", type=str, required=True, help="The path of the model path")
    parser.add_argument('-m',"--model_name", type=str, default='model.onnx',help="The name of the model used")    
    parser.add_argument('-fs',"--file_suffix", type=str,default="*.nii.gz", help="You want the suffix of the inference files")    
    parser.add_argument('-g',"--use_gpu", action="store_true",default=True, help="Whether to use a GPU for acceleration")    
    
    my_infer = Predictor(parser.parse_args())
    my_infer.run_infer()

if __name__ == '__main__':
    main_entry()