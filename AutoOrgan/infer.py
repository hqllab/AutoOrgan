import onnxruntime
import numpy as np
import cupy as cp
from tqdm import tqdm
from pathlib import Path

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

class OnnxInfer():
    def __init__(self,classes_count,onnx_path,use_gpu):
        self.classes_count = classes_count
        self.onnx_path = onnx_path
        self.use_gpu = use_gpu
        
        self.ort_session = self.load_model(onnx_path,use_gpu)
        
    def load_model(self,onnx_model_path,use_gpu):
        assert Path(onnx_model_path).exists(),'Onnx model is not found.'
        if not use_gpu:
            session = onnxruntime.InferenceSession(onnx_model_path,providers=["CPUExecutionProvider"])
            print("ONNX Runtime CPU 版本可用，正在使用 CPU 推理")
            return session
        try:
            available_providers = onnxruntime.get_available_providers()
            print("Available providers:", available_providers)

            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError("GPU is not available. CUDAExecutionProvider not found. Aborting inference.")
            
            session = onnxruntime.InferenceSession(onnx_model_path,providers=["CUDAExecutionProvider"])
            print("ONNX Runtime GPU 版本可用，正在使用 GPU 推理")
            return session
        
        except Exception as e:
            print("回退到 ONNX Runtime CPU 版本")
            session = onnxruntime.InferenceSession(onnx_model_path,providers=["CPUExecutionProvider"])
            return session

    def old_run_infer_ONNX(self,input_array,slicers,gaussian_array):
        assert Path(self.onnx_path).exists(),'Onnx model is not found.'
        
        gaussian_array = gaussian_array[None]
        predicted_logits = np.zeros(([self.classes_count] + list(input_array.shape)),dtype=np.float32)
        n_predictions = np.zeros((input_array.shape),dtype=np.float32)[None]
        
        for slicer in tqdm(slicers):
            input_array_part = input_array[slicer][None][None]
            input_array_part = np.ascontiguousarray(input_array_part, dtype=np.float32)
            ort_inputs = {self.ort_session.get_inputs()[0].name:input_array_part}
            ort_output = self.ort_session.run(None,ort_inputs)[0][0]
            
            predicted_logits[:,slicer[0],slicer[1],slicer[2]] += ort_output * gaussian_array 
            n_predictions[:,slicer[0],slicer[1],slicer[2]] += gaussian_array
        
        return predicted_logits / n_predictions
    
    def new_run_infer_ONNX(self,input_array,slicers,gaussian_array):
        
        # test_input_array = cp.array(input_array, dtype=cp.float32)
        # test_input_array = cp.ascontiguousarray(test_input_array)
        
        gaussian_array = cp.asarray(gaussian_array[None], dtype=np.float32)
        
        predicted_logits = cp.zeros(([self.classes_count] + list(input_array.shape)),dtype=np.float32)
        n_predictions = cp.zeros((input_array.shape),dtype=np.float32)[None]
        
        output_shape = (1, self.classes_count, 128, 128, 128)
        ort_output = cp.zeros(output_shape, dtype=cp.float32)
        ort_output = cp.ascontiguousarray(ort_output)        
        
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        io_binding = self.ort_session.io_binding()
        
        for slicer in tqdm(slicers):
            input_array_part = input_array[slicer][None][None]
            input_array_part = cp.array(input_array_part, dtype=cp.float32)
            # input_array_part = cp.ascontiguousarray(input_array_part)            
            
            # test_input_array_part = test_input_array[slicer][None][None]
            # test_input_array_part = cp.ascontiguousarray(test_input_array_part)            
            
            # result = cp.array_equal(input_array_part, test_input_array_part)
            # print(result)
            # abs_diff_sum = cp.sum(np.abs(input_array_part, test_input_array_part))
            # print("绝对值差之和:", abs_diff_sum)
            
            io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=cp.float32, shape=tuple(input_array_part.shape), buffer_ptr=input_array_part.data.ptr)
            io_binding.synchronize_inputs()
            io_binding.bind_output(name=output_name,device_type='cuda',device_id=0,element_type=np.float32,shape=output_shape,buffer_ptr=ort_output.data.ptr)
            
            self.ort_session.run_with_iobinding(io_binding) 
            ort_output_result = ort_output[0]
            
            predicted_logits[:,slicer[0],slicer[1],slicer[2]] += (ort_output_result * gaussian_array)
            n_predictions[:,slicer[0],slicer[1],slicer[2]] += gaussian_array
        
        for i in range(self.classes_count):
            predicted_logits[i] = predicted_logits[i] / n_predictions
        
        return predicted_logits.get()