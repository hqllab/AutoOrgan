import onnxruntime
import numpy as np
from tqdm import tqdm
from pathlib import Path
def load_model(onnx_model_path,use_gpu):
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

def run_infer_ONNX(basics_image,slicers,gaussian_array,classes_count,onnx_path,use_gpu):
    assert Path(onnx_path).exists(),'model_path is not exists!'
    
    input_array = basics_image.data
    gaussian_array = gaussian_array[None]
    predicted_logits = np.zeros(([classes_count] + list(input_array.shape)),dtype=np.float32)
    n_predictions = np.zeros((input_array.shape),dtype=np.float32)[None]
    
    ort_session = load_model(onnx_path,use_gpu)
    
    for slicer in tqdm(slicers):
        input_array_part = input_array[slicer][None][None]
        input_array_part = np.ascontiguousarray(input_array_part, dtype=np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name:input_array_part}
        ort_output = ort_session.run(None,ort_inputs)[0][0]
        
        predicted_logits[:,slicer[0],slicer[1],slicer[2]] += ort_output * gaussian_array 
        n_predictions[:,slicer[0],slicer[1],slicer[2]] += gaussian_array
    
    basics_image.infer_data = predicted_logits / n_predictions
    return basics_image