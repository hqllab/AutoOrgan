import cupyx.scipy.ndimage
import cupy as cp
import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from pathlib import Path

class OnnxInfer():
    def __init__(self,classes_count,onnx_path,use_gpu):
        self.classes_count = classes_count
        self.onnx_path = onnx_path
        self.use_gpu = use_gpu
        
        self.ort_session = self.load_model(onnx_path,use_gpu)
        
    def load_model(self,onnx_model_path,use_gpu):
        assert Path(onnx_model_path).exists(),'Onnx model is not found.'
        if not use_gpu:
            session = ort.InferenceSession(onnx_model_path,providers=["CPUExecutionProvider"])
            print("ONNX Runtime CPU 版本可用，正在使用 CPU 推理")
            return session
        try:
            available_providers = ort.get_available_providers()
            print("Available providers:", available_providers)

            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError("GPU is not available. CUDAExecutionProvider not found. Aborting inference.")
            
            session = ort.InferenceSession(onnx_model_path,providers=["CUDAExecutionProvider"])
            print("ONNX Runtime GPU 版本可用，正在使用 GPU 推理")
            return session
        
        except Exception as e:
            print("回退到 ONNX Runtime CPU 版本")
            session = ort.InferenceSession(onnx_model_path,providers=["CPUExecutionProvider"])
            return session

    def run_infer_ONNX(self,input_array,slicers,gaussian_array,parameters_dict):
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
            input_array_part = cp.ascontiguousarray(input_array_part)            
            
            io_binding.bind_input(name=input_name, device_type='cuda', device_id=0, element_type=cp.float32, shape=tuple(input_array_part.shape), buffer_ptr=input_array_part.data.ptr)
            io_binding.synchronize_inputs()
            io_binding.bind_output(name=output_name,device_type='cuda',device_id=0,element_type=np.float32,shape=output_shape,buffer_ptr=ort_output.data.ptr)
            
            self.ort_session.run_with_iobinding(io_binding) 
            ort_output_result = ort_output[0]
            
            predicted_logits[:,slicer[0],slicer[1],slicer[2]] += (ort_output_result * gaussian_array)
            n_predictions[:,slicer[0],slicer[1],slicer[2]] += gaussian_array
        
        for i in range(self.classes_count):
            predicted_logits[i] = predicted_logits[i] / n_predictions

        del gaussian_array, n_predictions
        cp.get_default_memory_pool().free_all_blocks()
        
        slicer_revert_padding = parameters_dict.pad_bbox
        predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding])]
        predicted_logits = cp.argmax(predicted_logits,0)
        
        zoom_factors = [t / o for t, o in zip(parameters_dict.shape_after_crop, predicted_logits.shape)]
        predicted_logits = cupyx.scipy.ndimage.zoom(predicted_logits,zoom_factors,order=0)
        
        slicer = tuple([slice(*i) for i in parameters_dict.crop_bbox])
        segmentation_reverted_cropping = np.zeros(parameters_dict.origin_shape,dtype=np.uint16)
        segmentation_reverted_cropping[slicer] = predicted_logits.get()
        image_data = segmentation_reverted_cropping.astype(np.uint8)
        return image_data