import os
import qt
import vtk
import json
import time
import shutil
import logging
import subprocess
import numpy as np
import SimpleITK as sitk
from typing import Annotated
from pathlib import Path
from functools import lru_cache

import slicer
import qSlicerSegmentationsModuleWidgetsPythonQt
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper,WithinRange
from slicer import vtkMRMLScalarVolumeNode

def data_type_convert(str_or_dt):
    if isinstance(str_or_dt, str):
        if len(str_or_dt) > 1:
            return str_or_dt
        assert 0, f'invalid str type: {type(str_or_dt)}!'
        
    elif isinstance(str_or_dt, np.dtype):
        [(_,dt)] = str_or_dt.descr
        # print(f'date type: {dt}')
        return dt
    else:
        assert 0,f'unrecognized type: {str_or_dt}'

class BackendImage(object):
    def __init__(self, data, meta=None):
        self.init_meta(meta)
        self.init_data(data)

    def init_meta(self,meta):
        from munch import DefaultMunch
        if isinstance(meta, DefaultMunch):
            self.meta_dict = meta
        elif isinstance(meta, dict):
            self.meta_dict = DefaultMunch.fromDict(meta)
        elif isinstance(meta, bytes):
            self.meta_dict = DefaultMunch.fromDict(json.loads(meta))
        elif meta is None: 
            print('meta auto init')
        else:
            assert f'error meta data type: {type(meta)}!'
            
    def init_data(self,data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, bytes):
            data_type = self.meta_dict.DataType
            dtype=data_type_convert(data_type)
            shape_x, shape_y, shape_z = self.meta_dict.Shape.X, self.meta_dict.Shape.Y, self.meta_dict.Shape.Z
            self.data = np.frombuffer(data, dtype).reshape(shape_z,shape_y, shape_x).astype(np.float32)
        elif data == None:
            assert 'data is None!'
        else:
            assert f'error data type: {type(data)}!'

def load_model(onnx_model_path,use_gpu):
    import onnxruntime
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

class EmitStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        # 去除换行符并追加到文本框
        if text.strip():
            self.text_edit.appendPlainText(text.strip())

    def flush(self):
        pass

def run_infer_ONNX(basics_image,slicers,gaussian_array,classes_count,onnx_path,use_gpu,logCallback=None):
    from tqdm import tqdm
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

def create_nonzero_mask(data):
    from scipy.ndimage import binary_fill_holes
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = data != 0
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask

def bounding_box_to_slice(bounding_box):
    return tuple([slice(*i) for i in bounding_box])

def get_bbox_from_mask(mask: np.ndarray):
    
    Z, X, Y = mask.shape
    minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
    zidx = list(range(Z))
    for z in zidx:
        if np.any(mask[z]):
            minzidx = z
            break
    for z in zidx[::-1]:
        if np.any(mask[z]):
            maxzidx = z + 1
            break

    xidx = list(range(X))
    for x in xidx:
        if np.any(mask[:, x]):
            minxidx = x
            break
    for x in xidx[::-1]:
        if np.any(mask[:, x]):
            maxxidx = x + 1
            break

    yidx = list(range(Y))
    for y in yidx:
        if np.any(mask[:, :, y]):
            minyidx = y
            break
    for y in yidx[::-1]:
        if np.any(mask[:, :, y]):
            maxyidx = y + 1
            break
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

def crop_to_nonzero(data):
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    data_shape = data.shape
    data = data[tuple([*slicer])]
    crop_list = [[bbox[0][0],data_shape[0]-bbox[0][1]],[bbox[1][0],data_shape[1]-bbox[1][1]],[bbox[2][0],data_shape[2]-bbox[2][1]]]
    return data,bbox,crop_list

def ct_znorm(img3d, properties):
    infos = properties['0']
    mean_intensity = infos['mean']
    std_intensity = infos['std']
    lower_bound = infos['percentile_00_5']
    upper_bound = infos['percentile_99_5']
    ret_img = np.clip(img3d, lower_bound, upper_bound)
    ret_img = (ret_img - mean_intensity) / max(std_intensity, 1e-8)
    return ret_img

def compute_new_shape(old_shape,old_spacing,new_spacing):
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape

def get_lowres_axis(new_spacing):
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0] 
    return axis

def get_do_separate_z(spacing,  anisotropy_threshold=3):
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z

def resample_data_or_seg(data, new_shape,is_seg, axis, order = 3, do_separate_z = False, order_z = 0):
    from skimage.transform import resize
    from scipy.ndimage import map_coordinates
    resize_fn = resize
    kwargs = {'mode': 'edge', 'anti_aliasing': False}
    dtype_data = data.dtype
    shape = np.array(data.shape)
    new_shape = np.array(new_shape)
    
    if np.any(shape != new_shape):
        data = data.astype(float)
        
        if do_separate_z:
            assert len(axis) == 1, "only one anisotropic axis supported"
            axis = axis[0]
            if axis == 0: new_shape_2d = new_shape[1:]
            elif axis == 1: new_shape_2d = new_shape[[0, 2]]
            else: new_shape_2d = new_shape[:-1]

            reshaped_final_data = []
            reshaped_data = []
            for slice_id in range(shape[axis]):
                if axis == 0: reshaped_data.append(resize_fn(data[slice_id], new_shape_2d, order, **kwargs))
                elif axis == 1: reshaped_data.append(resize_fn(data[:, slice_id], new_shape_2d, order, **kwargs))
                else: reshaped_data.append(resize_fn(data[:, :, slice_id], new_shape_2d, order, **kwargs))
                
            reshaped_data = np.stack(reshaped_data, axis)
            
            if shape[axis] != new_shape[axis]:

                # The following few lines are blatantly copied and modified from sklearn's resize()
                rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                orig_rows, orig_cols, orig_dim = reshaped_data.shape

                row_scale = float(orig_rows) / rows
                col_scale = float(orig_cols) / cols
                dim_scale = float(orig_dim) / dim

                map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                map_rows = row_scale * (map_rows + 0.5) - 0.5
                map_cols = col_scale * (map_cols + 0.5) - 0.5
                map_dims = dim_scale * (map_dims + 0.5) - 0.5

                coord_map = np.array([map_rows, map_cols, map_dims])
                if not is_seg or order_z == 0: reshaped_final_data.append(map_coordinates(reshaped_data, coord_map, order=order_z,mode='nearest')[None])
                else:
                    unique_labels = np.sort(np.unique(reshaped_data))  # np.unique(reshaped_data)
                    reshaped = np.zeros(new_shape, dtype=dtype_data)

                    for i, cl in enumerate(unique_labels):
                        reshaped_multihot = np.round(map_coordinates((reshaped_data == cl).astype(float), coord_map, order=order_z,
                                            mode='nearest'))
                        reshaped[reshaped_multihot > 0.5] = cl
                    reshaped_final_data.append(reshaped[None])
            else: reshaped_final_data.append(reshaped_data[None])
                
            reshaped_final_data = np.vstack(reshaped_final_data)
        else: reshaped_final_data = resize_fn(data, new_shape, order, **kwargs)
        if do_separate_z:
            return reshaped_final_data.astype(dtype_data)[0]
        else:
            return reshaped_final_data.astype(dtype_data)
    else:
        print("no resampling necessary")
        return data

def resample_data_or_seg_to_shape(data,new_shape,current_spacing,new_spacing,is_seg = False,order= 3, order_z = 0,force_separate_z = False,separate_z_anisotropy_threshold= 3):
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z: axis = get_lowres_axis(current_spacing)
        else: axis = None
    else:
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None

    if axis is not None:
        if len(axis) == 3: do_separate_z = False
        elif len(axis) == 2: do_separate_z = False
        else: pass

    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped

def padding(image, patch_size):
    new_shape = patch_size
    if len(patch_size) < len(image.shape):
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)
        
    old_shape = np.array(image.shape)
    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]
    
    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]
    
    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        result_array = np.pad(image, pad_list, 'constant')
    else:
        result_array = image
        
    pad_array = np.array(pad_list)
    pad_array[:, 1] = np.array(result_array.shape) - pad_array[:, 1]
    slicer = tuple(slice(*i) for i in pad_array)
    pad_list = np.array(pad_list).ravel().tolist()
    return result_array, slicer, pad_list

def compute_steps_for_sliding_window(image_size, tile_size, tile_step_size):
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)
    
    slicers = []
    for sx in steps[0]:
        for sy in steps[1]:
            for sz in steps[2]:
                slicers.append(tuple([*[slice(si, si + ti) for si, ti in zip((sx, sy, sz), tile_size)]]))
    return slicers

@lru_cache(maxsize=2)
def compute_gaussian(tile_size, sigma_scale,value_scaling_factor, dtype=np.float16 ):
    from scipy.ndimage import gaussian_filter
    temporary_array = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    temporary_array[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(temporary_array, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * value_scaling_factor
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(gaussian_importance_map[gaussian_importance_map != 0])
    return gaussian_importance_map

def padding(image, patch_size):
    new_shape = patch_size
    if len(patch_size) < len(image.shape):
        new_shape = list(image.shape[:len(image.shape) - len(new_shape)]) + list(new_shape)
        
    old_shape = np.array(image.shape)
    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]
    
    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [list(i) for i in zip(pad_below, pad_above)]
    
    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        result_array = np.pad(image, pad_list, 'constant')
    else:
        result_array = image
        
    pad_array = np.array(pad_list)
    pad_array[:, 1] = np.array(result_array.shape) - pad_array[:, 1]
    slicer = tuple(slice(*i) for i in pad_array)
    pad_list = np.array(pad_list).ravel().tolist()
    return result_array, slicer, pad_list

def preprocess(image,config_dict):
    from munch import DefaultMunch
    
    plans_dict = DefaultMunch.fromDict(config_dict)
    
    parameters_dict = DefaultMunch()
    
    data = image.data # z,y,x
    parameters_dict.origin_shape = data.shape
    
    cropped_data,crop_bbox,crop_list = crop_to_nonzero(data) 
    parameters_dict.shape_after_crop = cropped_data.shape
    parameters_dict.crop_bbox = crop_bbox
    parameters_dict.crop_list = crop_list
    
    cropped_normed_data = ct_znorm(cropped_data,plans_dict.foreground_intensity_properties_per_channel)
    
    target_spacing = plans_dict.original_median_spacing_after_transp
    original_spacing = (abs(image.meta_dict.PixelSize.Z),image.meta_dict.PixelSize.Y,image.meta_dict.PixelSize.X)
    new_shape = compute_new_shape(cropped_normed_data.shape, original_spacing, target_spacing)
    
    order = plans_dict.configurations['3d_fullres'].resampling_fn_probabilities_kwargs.order
    order_z = plans_dict.configurations['3d_fullres'].resampling_fn_probabilities_kwargs.order_z
    force_separate_z = plans_dict.configurations['3d_fullres'].resampling_fn_probabilities_kwargs.force_separate_z
    cropped_normed_resampled_data = resample_data_or_seg_to_shape(cropped_normed_data,new_shape,original_spacing,target_spacing,order=order,order_z=order_z,force_separate_z=force_separate_z)
    parameters_dict.before_preprocess_spacing = original_spacing
    parameters_dict.after_preprocess_spacing = target_spacing
    parameters_dict.shape_after_crop_resample = cropped_normed_resampled_data.shape
    
    patch_size = plans_dict.configurations['3d_fullres'].patch_size
    cropped_normed_resampled_patched_data,pad_bbox,pad_list  = padding(cropped_normed_resampled_data,patch_size)
    slicers = compute_steps_for_sliding_window(cropped_normed_resampled_patched_data.shape,patch_size,0.5)
    gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1. / 8,value_scaling_factor=10)
    parameters_dict.pad_bbox = pad_bbox
    parameters_dict.pad_list = pad_list
    parameters_dict.patch_size = patch_size
    parameters_dict.shape_after_crop_resample_pad = cropped_normed_resampled_patched_data.shape
    
    image.data = cropped_normed_resampled_patched_data
    
    return image,slicers,gaussian,parameters_dict

def infer(receive_dict,config_dict,model_path,use_gpu,classes_count,logCallback):
    from munch import DefaultMunch
    receive_dict = DefaultMunch.fromDict(receive_dict)
    
    ct_data = receive_dict.InImage.CTImage.Data
    ct_meta = receive_dict.InImage.CTImage.Meta 
    ct_image = BackendImage(ct_data, ct_meta)
    ct_image.flag = 'ct'
        
    ct_image,slicers,gaussian,ct_parameters_dict = preprocess(ct_image, config_dict)
    ct_image = run_infer_ONNX(ct_image,slicers,gaussian,classes_count,model_path,use_gpu,logCallback)
    
    ct_image = postprocess(ct_image,ct_parameters_dict)
    # ct_image = work(ct_image,labels_dict)
    
    return ct_image

def postprocess(image, parameters_dict):
    from skimage.transform import resize
    image_data = image.infer_data
    
    slicer_revert_padding = parameters_dict.pad_bbox
    crop_image_data = image_data[tuple([slice(None), *slicer_revert_padding])]
    crop_image_data = np.argmax(crop_image_data,0)
    
    crop_zoom_image_data = resize(crop_image_data,parameters_dict.shape_after_crop,order=0)
    
    slicer = tuple([slice(*i) for i in parameters_dict.crop_bbox])
    segmentation_reverted_cropping = np.zeros(parameters_dict.origin_shape,dtype=np.uint16)
    segmentation_reverted_cropping[slicer] = crop_zoom_image_data
    image_data = segmentation_reverted_cropping
    
    image.infer_data = image_data.astype(np.uint8)
    return image

def entry_main(input_path,output_path,model_path,logCallback,use_gpu=False):
    model_path = Path(model_path)
    onnx_path_file_list = list(model_path.glob('*.onnx'))
    if len(list(onnx_path_file_list)) == 0:
        raise Exception('model path must include onnx file')
    if (model_path / 'dataset.json').exists() == False:
        raise Exception('model path must include dataset.json')
    if (model_path / 'plans.json').exists() == False:
        raise Exception('model path must include plans.json')
    
    with open(model_path / 'dataset.json') as f:
        dataset_dict = json.load(f)
    class_count = len(dataset_dict['labels'])
    
    with open(model_path / 'plans.json') as f:
        config_dict = json.load(f)
    
    ct_image = sitk.ReadImage(input_path)
    ct_x_spacing,ct_y_spacing,ct_z_spacing = ct_image.GetSpacing()
    ct_array = sitk.GetArrayFromImage(ct_image)     
    ct_z,ct_y,ct_x = ct_array.shape   
     
    ct_bytes = ct_array.tobytes()
    
    parameters_dict = {
        "InImage": {
            "CTImage":{
                "Data": ct_bytes,
                "Meta": {
                    "DataType":ct_array.dtype.descr[0][1],
                    "PixelSize":{"X":ct_x_spacing,"Y":ct_y_spacing,"Z":ct_z_spacing},
                    "Shape":{
                        "X":ct_x,
                        "Y":ct_y,
                        "Z":ct_z
                    }
            }
            }
        }
        }
    
    infer_result = infer(parameters_dict,config_dict,onnx_path_file_list[0],use_gpu,class_count,logCallback)
    infer_array = infer_result.infer_data
    image = sitk.GetImageFromArray(infer_array)
    image.SetSpacing((ct_x_spacing,ct_y_spacing,ct_z_spacing))
    image.SetOrigin(ct_image.GetOrigin())
    image.SetDirection(ct_image.GetDirection())
    sitk.WriteImage(image,output_path)
    
# def main():
    # parser = argparse.ArgumentParser(description="AutoOrgan")

    # parser.add_argument('-i', type=str, required=True, help='input path')
    # parser.add_argument('-o', type=str,  help='output path')
    # parser.add_argument('-m', type=str,  help='model path,it must include onnx and dataset.json and plans.json')
    # parser.add_argument('-g', action='store_true', help='use gpu')

    # args = parser.parse_args()
    # entry_main(args.i,args.o,args.m,args.g)

#
# AutoOrganSlicer
#

class AutoOrganSlicer(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("AutoOrganSlicer")  
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = [] 
        self.parent.contributors = ["Wang Nannan"]  
        self.parent.helpText = ''
        self.parent.acknowledgementText = ''

#
# AutoOrganSlicerParameterNode
#

@parameterNodeWrapper
class AutoOrganSlicerParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode

#
# AutoOrganSlicerWidget
#

opacity_value = 0

class CloseApplicationEventFilter(qt.QWidget):
  def eventFilter(self, object, event):
    global opacity_value
    if event.type() == qt.QEvent.KeyPress:
        key = event.key()
        if key == ord('A'):  # 按下 'A' 键
            slice_widget = slicer.app.layoutManager().sliceWidget("Red")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(0)            
            
            slice_widget = slicer.app.layoutManager().sliceWidget("Green")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(0)    
            
            slice_widget = slicer.app.layoutManager().sliceWidget("Yellow")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(0)   
        elif key == ord('S'):  # 按下 'S' 键
            slice_widget = slicer.app.layoutManager().sliceWidget("Red")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(opacity_value)            
            
            slice_widget = slicer.app.layoutManager().sliceWidget("Green")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(opacity_value)    
            
            slice_widget = slicer.app.layoutManager().sliceWidget("Yellow")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(opacity_value)   
        return True  # 表示事件已被处理
    return False

class AutoOrganSlicerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self) 
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False
        self.parameterSetNode = None
        # slicer.app.pythonConsole().clear()
        self.model_path = None
        
    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AutoOrganSlicer.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = AutoOrganSlicerLogic()
        self.logic.logCallback = self.addLog

        self.key_press_filter = CloseApplicationEventFilter()
        slicer.util.mainWindow().installEventFilter(self.key_press_filter)
        
        # tasks = ["骨骼","器官"]
        # for task in tasks:
        #     self.ui.taskComboBox.addItem(task,task)
        
        self.color_dict = {
            "灰度图":"vtkMRMLColorTableNodeGrey",
            'PET-Rainbow2':"vtkMRMLPETProceduralColorNodePET-Rainbow2",
            'fMRI彩图':"vtkMRMLColorTableNodefMRI",
            'PET-Heat':"vtkMRMLPETProceduralColorNodePET-Heat",
        }
        
        for color in self.color_dict.keys():
            self.ui.colorComboBox.addItem(color,color)
        
        self.editor = qSlicerSegmentationsModuleWidgetsPythonQt.qMRMLSegmentEditorWidget()
        self.editor.setMaximumNumberOfUndoStates(10)
        self.selectParameterNode()
        self.editor.setMRMLScene(slicer.mrmlScene)
        self.ui.segmentButton.layout().addWidget(self.editor)

        self.subjectHierarchyTreeView = slicer.qMRMLSubjectHierarchyTreeView()
        self.subjectHierarchyTreeView.setMRMLScene(slicer.mrmlScene)
        self.subjectHierarchyTreeView.setColumnHidden(self.subjectHierarchyTreeView.model().idColumn, True)
        self.subjectHierarchyTreeView.setColumnHidden(self.subjectHierarchyTreeView.model().colorColumn, True)
        self.subjectHierarchyTreeView.setColumnHidden(self.subjectHierarchyTreeView.model().transformColumn, True)
        self.subjectHierarchyTreeView.model().setHorizontalHeaderLabels(['所有数据', ...])
        self.ui.dataButton.layout().addWidget(self.subjectHierarchyTreeView)
        
        segmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
        if not segmentationNode:
            segmentationNode = slicer.vtkMRMLSegmentationNode()
            segmentationNode.SetName("DefaultSegmentation")
            slicer.mrmlScene.AddNode(segmentationNode)

        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(slicer.mrmlScene.GetNodeByID(self.ui.inputVolumeSelector.currentNodeID))
        self.editor.setSegmentationNode(segmentationNode)
        
        self.editor.setSegmentationNode(segmentationNode)
        self.editor.show()        

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        
        # 主要应用
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        # self.ui.taskComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.colorComboBox.currentTextChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.gpuCheckBox.connect('toggled(bool)', self.updateParameterNodeFromGUI)

        # 内部工具
        self.ui.checkButton.connect('clicked(bool)', self.onCheckEnvironment)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.dicomButton.connect("clicked(bool)", self.onDicomButton)
        self.ui.modelButton.connect("clicked(bool)", self.onModelButton)
        self.ui.saveSegmentButton.connect("clicked(bool)", self.onSaveSegmentButton)
        self.ui.autoWindowButton.connect('clicked(bool)',self.onAutoWindowButton)
        
        self.ui.levelEdit.textChanged.connect(self.onWindowChange)
        self.ui.widthEdit.textChanged.connect(self.onWindowChange)
        self.ui.opacitySpinBox.textChanged.connect(self.onOpacityChange)
        self.ui.colorComboBox.currentIndexChanged.connect(self.onColorChange)
        
        self.initializeParameterNode()
    
    def set_window_level_from_dicom(self,volume_node):
        display_node = volume_node.GetDisplayNode()
        
        # 获取当前体积节点的第一个切片的 Instance UID
        instance_uid = volume_node.GetAttribute("DICOM.instanceUIDs").split()
        if not instance_uid:
            print("No DICOM.InstanceUID found in display node.")
            return
        
        image_class = volume_node.GetAttribute("DICOM.instanceUIDs")
        
        # 使用 slicer.dicomDatabase 获取该实例对应的文件路径
        file_path = slicer.dicomDatabase.fileForInstance(instance_uid[0])
        if not file_path:
            print(f"No file found for Instance UID {instance_uid}")
            return
        
        # 使用 DICOMUtils 读取 DICOM 文件
        import pydicom
        try:
            dcm_data = pydicom.dcmread(file_path)
        except Exception as e:
            print(f"Error reading DICOM file: {e}")
            return

        # 获取 Window Width (0028,1150) 和 Window Center (0028,1151)
        window_width = dcm_data.WindowWidth
        window_center = dcm_data.WindowCenter
        image_class = dcm_data.Modality

        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)

        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)

        if image_class == "CT":
            # CT 通常使用 400 作为窗宽，-1500 作为窗位

            window_center = 400
            window_width = 1500
            display_node.AutoWindowLevelOff()
            display_node.SetWindow(window_width)
            display_node.SetLevel(window_center)

            self.ui.widthEdit.setValue(window_width)
            self.ui.levelEdit.setValue(window_center)
        
        else:
            display_node.AutoWindowLevelOff()
            display_node.SetWindow(window_width)
            display_node.SetLevel(window_center)

            self.ui.widthEdit.setValue(window_width)
            self.ui.levelEdit.setValue(window_center)

        # self.logic.log(f"Set Window Level from DICOM: WW={window_width}, WC={window_center}")
        
    def selectParameterNode(self):
        # Select parameter set node if one is found in the scene, and create one otherwise
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        if self.parameterSetNode == segmentEditorNode:
            # nothing changed
            return
        self.parameterSetNode = segmentEditorNode
        self.editor.setMRMLSegmentEditorNode(self.parameterSetNode)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)
            
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.ui.inputVolumeSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
        task = self._parameterNode.GetParameter("Task")
        
        # self.ui.taskComboBox.setCurrentIndex(self.ui.taskComboBox.findData(task))
        # self.ui.gpuCheckBox.checked = False
        self.ui.gpuCheckBox.checked =  self._parameterNode.GetParameter("Device") == "True"

        # Update buttons states and tooltips
        inputVolume = self._parameterNode.GetNodeReference("InputVolume")
        if inputVolume:
            self.ui.applyButton.toolTip = "Start segmentation"
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = "Select input volume"
            self.ui.applyButton.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        
        wasModified = self._parameterNode.StartModify()
        
        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputVolumeSelector.currentNodeID)
        # self._parameterNode.SetParameter("Task", self.ui.taskComboBox.currentData)
        self._parameterNode.SetParameter("Device", "GPU" if self.ui.gpuCheckBox.checked else "CPU")
        
        self._parameterNode.EndModify(wasModified)
        
        volumeNode = self.ui.inputVolumeSelector.currentNode()
        # print(volumeNode)
        self.set_window_level_from_dicom(volumeNode)

    def addLog(self, text):
        """Append text to log window
        """
        self.ui.statusLabel.appendPlainText(text)
        slicer.app.processEvents()  # force update

    def onApplyButton(self) -> None:
        self.ui.statusLabel.plainText = ''
        
        with slicer.util.tryWithErrorDisplay(("分割失败."), waitCursor=True):
            outputSegmentationFile,tempFolder = self.logic.process(self.ui.inputVolumeSelector.currentNode(),self.ui.gpuCheckBox.checked)
        self.ui.statusLabel.appendPlainText("\n分割完成。")
        
        segmentation_node = slicer.util.loadSegmentation(outputSegmentationFile)
        segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(slicer.mrmlScene.GetNodeByID(self.ui.inputVolumeSelector.currentNodeID))
        self.editor.setSegmentationNode(segmentation_node)
        self.editor.show()
        
        self.logic.log("\n清空暂存区...")
        if os.path.isdir(tempFolder):
            shutil.rmtree(tempFolder)
        
    def onCheckEnvironment(self):
        import qt
        
        # if self.ui.gpuCheckBox.checked:
            # if not self.logic.check_nvidia_smi():
                # qt.QMessageBox.warning(slicer.util.mainWindow(), "Warning", "Please install NVIDIA driver and CUDA Toolkit first.")
                # self.ui.gpuCheckBox.checked = False
        
        try:
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            self.logic.stupPythonRequirements()
            slicer.app.restoreOverrideCursor()
            self.ui.statusLabel.appendPlainText(f"环境检查完成，可以运行。")
            
        except Exception as e:
            slicer.app.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            self.ui.statusLabel.appendPlainText(f"安装Python依赖失败，请检查。:\n{e}\n")
            
            if isinstance(e, InstallError):
                restartRequired = e.restartRequired
                
            if restartRequired:
                self.ui.statusLabel.appendPlainText("\n3D slicer需要重启，请重启。")
                if slicer.util.confirmOkCancelDisplay(
                    "3D slicer需要重启，请重启.\n按下回车键确认重启。",
                    "确认重启。",
                    detailedText=str(e)
                ):
                    slicer.util.restart()
                else:
                    return
            else:
                slicer.util.errorDisplay(f"安装失败。\n\n{e}")
                return
            
    def onDicomButton(self):
        import qt
        try:
            default_path = str(Path.home() / "Desktop" / "Slicer.nii.gz")
            # self.logic.log(f"Writing input file to {self.ui.inputVolumeSelector.currentNode()}")
            
            qfiledialog = qt.QFileDialog()
            save_path = qfiledialog.getSaveFileName(None,'选择保存路径',default_path,"NIfTI Files (*.nii *.nii.gz)")
            
            if save_path == "":
                return
            
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            
            inputVolume = self.ui.inputVolumeSelector.currentNode()
            volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
            volumeStorageNode.SetFileName(save_path)
            volumeStorageNode.UseCompressionOn()
            volumeStorageNode.WriteData(inputVolume)
            volumeStorageNode.UnRegister(None)
            
            slicer.app.restoreOverrideCursor()
            self.ui.statusLabel.appendPlainText(f"Dicom2Nii is done.")
            
        except Exception as e:
            slicer.app.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            self.ui.statusLabel.appendPlainText(f"Dicom2Nii is failure.")
    
    def onModelButton(self):
        import qt
        try:
            default_path = str(Path.home() / "Desktop")
            qfiledialog = qt.QFileDialog()
            self.logic.model_path = qfiledialog.getExistingDirectory(None,'选择模型路径',default_path)
            
            if self.logic.model_path == "":
                return
            self.ui.statusLabel.appendPlainText(f"Choose is done.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.ui.statusLabel.appendPlainText(f"Choose is failure.")    
    
    
    def onSaveSegmentButton(self):
        import qt
        
        try:
            default_path = str(Path.home() / "Desktop" / "SlicerSegment.nii")
            
            qfiledialog = qt.QFileDialog()
            save_path = qfiledialog.getSaveFileName(None,'Choose save path',default_path,"NIfTI Files (*.nii *.nii.gz)")
            
            if save_path == "":
                return
            
            slicer.app.setOverrideCursor(qt.Qt.WaitCursor)
            self.logic.log(f"Writing segment result file to {save_path}")
            labelmap_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")

            slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(
                self.editor.segmentationNode(),
                labelmap_node,
                slicer.vtkSegmentation.EXTENT_REFERENCE_GEOMETRY)
            
            slicer.util.saveNode(labelmap_node, save_path)
            slicer.mrmlScene.RemoveNode(labelmap_node)
            slicer.app.restoreOverrideCursor()
            
        except Exception as e:
            slicer.app.restoreOverrideCursor()
            import traceback
            traceback.print_exc()
            self.ui.statusLabel.appendPlainText(f"Save segment result is failure.")

    def onWindowChange(self):
        if self.ui.widthEdit.text == "" or self.ui.levelEdit.text == "":
            return
        try:
            width_value = int(self.ui.widthEdit.text)
            level_value = int(self.ui.levelEdit.text)
            volumeNode = self.ui.inputVolumeSelector.currentNode()
            displayNode = volumeNode.GetDisplayNode()
            displayNode.AutoWindowLevelOff()
            displayNode.SetWindow(width_value)
            displayNode.SetLevel(level_value)
            
        except Exception as e:
            self.logic.log(f"输入错误.")

    def onAutoWindowButton(self):
        volumeNode = self.ui.inputVolumeSelector.currentNode()
        displayNode = volumeNode.GetDisplayNode()
        displayNode.AutoWindowLevelOn()

        current_window_value = displayNode.GetWindow()
        current_level_value = displayNode.GetLevel()
        self.ui.widthEdit.setValue(current_window_value)
        self.ui.levelEdit.setValue(current_level_value)
        
    def onOpacityChange(self):
        global opacity_value
        if self.ui.opacitySpinBox.text == "":
            return
        try:
            opacity_value = float(self.ui.opacitySpinBox.text)
            # volumeNode = self.ui.inputVolumeSelector.currentNode()
            # displayNode = volumeNode.GetDisplayNode()
            # displayNode.SetOpacity(opacity_value)
            # displayNode.AutoWindowLevelOff()
            
            slice_widget = slicer.app.layoutManager().sliceWidget("Red")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(opacity_value)            
            
            slice_widget = slicer.app.layoutManager().sliceWidget("Green")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(opacity_value)    
            
            slice_widget = slicer.app.layoutManager().sliceWidget("Yellow")
            slice_composite_node = slice_widget.sliceLogic().GetSliceCompositeNode()
            slice_composite_node.SetForegroundOpacity(opacity_value)                            
            
        except Exception as e:
            self.logic.log(f"输入错误.{e}")
      
    def onColorChange(self):
        color_name = self.ui.colorComboBox.currentText
        
        volumeNode = self.ui.inputVolumeSelector.currentNode()
        if volumeNode is None:
            return
        
        displayNode = volumeNode.GetDisplayNode()
        displayNode.AutoWindowLevelOff()
        displayNode.SetAndObserveColorNodeID(self.color_dict[color_name])         
        
class InstallError(Exception):
    def __init__(self, message, restartRequired=False):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.message = message
        self.restartRequired = restartRequired
    def __str__(self):
        return self.message
    
class AutoOrganSlicerLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.clearOutputFolder = True
        self.logCallback = None
        self.model_path = ''
        self.AutoOrganPythonPackageDownloadUrl = ""  

    def pipInstallSelective(self, packageToInstall, installCommand, packagesToSkip):
        slicer.util.pip_install(f"AutoOrgan")
        # slicer.util.pip_install(f"{installCommand} --no-deps")
        # skippedRequirements = []  # list of all missed packages and their version

        # import importlib.metadata
        # metadataPath = [p for p in importlib.metadata.files(packageToInstall) if 'METADATA' in str(p)][0]
        # metadataPath.locate()

        # # Remove line: `Requires-Dist: SimpleITK (==2.0.2)`
        # # User Latin-1 encoding to read the file, as it may contain non-ASCII characters and not necessarily in UTF-8 encoding.
        # filteredMetadata = ""
        # with open(metadataPath.locate(), "r+", encoding="latin1") as file:
        #     for line in file:
        #         skipThisPackage = False
        #         requirementPrefix = 'Requires-Dist: '
        #         if line.startswith(requirementPrefix):
        #             for packageToSkip in packagesToSkip:
        #                 if packageToSkip in line:
        #                     skipThisPackage = True
        #                     break
        #         if skipThisPackage:
        #             skippedRequirements.append(line.removeprefix(requirementPrefix))
        #             continue
        #         filteredMetadata += line
        #     # Update file content with filtered result
        #     file.seek(0)
        #     file.write(filteredMetadata)
        #     file.truncate()

        # # Install all dependencies but the ones listed in packagesToSkip
        # import importlib.metadata
        # requirements = importlib.metadata.requires(packageToInstall)
        # for requirement in requirements:
        #     skipThisPackage = False
        #     for packageToSkip in packagesToSkip:
        #         if requirement.startswith(packageToSkip):
        #             # Do not install
        #             skipThisPackage = True
        #             break

        #     match = False
        #     if not match:
        #         # Rewrite optional depdendencies info returned by importlib.metadata.requires to be valid for pip_install:
        #         # Requirement Original: ruff; extra == "dev"
        #         # Requirement Rewritten: ruff
        #         match = re.match(r"([\S]+)[\s]*; extra == \"([^\"]+)\"", requirement)
        #         if match:
        #             requirement = f"{match.group(1)}"
        #     if not match:
        #         # nibabel >=2.3.0 -> rewrite to: nibabel>=2.3.0
        #         match = re.match("([\S]+)[\s](.+)", requirement)
        #         if match:
        #             requirement = f"{match.group(1)}{match.group(2)}"

        #     if skipThisPackage:
        #         self.log(f'- Skip {requirement}')
        #     else:
        #         self.log(f'- Installing {requirement}...')
        #         slicer.util.pip_install(requirement)

        # return skippedRequirements

    def stupPythonRequirements(self):
        needToInstallSegmenter = False
        
        packagesToSkip = [
            'SimpleITK'  # Slicer's SimpleITK uses a special IO class, which should not be replaced
            ]
        try:
            import numpy as np
        except ModuleNotFoundError:
            slicer.util.pip_install("numpy")
        try:
            import onnxruntime
        except ModuleNotFoundError:
            slicer.util.pip_install("onnxruntime-gpu")

        try:
            from tqdm import tqdm
        except ModuleNotFoundError:
            slicer.util.pip_install("tqdm")

        try:
            from munch import DefaultMunch
        except ModuleNotFoundError:
            slicer.util.pip_install("munch")

        try:
            from skimage.transform import resize
        except ModuleNotFoundError:
            slicer.util.pip_install("scikit-image")
            
        try:
            from scipy.ndimage import map_coordinates,gaussian_filter
        except ModuleNotFoundError:
            slicer.util.pip_install("scipy")
        
        if needToInstallSegmenter:
            self.log(f'正在安装AutoOrgan，请等待...')
            self.pipInstallSelective("AutoOrgan",self.AutoOrganPythonPackageDownloadUrl,packagesToSkip)
        
    def log(self, text):
        logging.info(text)
        if self.logCallback:
            self.logCallback(text)

    def check_nvidia_smi(self):
        driver_version = None
        cuda_version = None
        gpu_model = None
        
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='GBK',
                text=True,
                check=True
            )
        except Exception as e:
            self.log(f"Failed to run nvidia-smi: {e}")
            return False
        lines = result.stdout.split("\n")
        for line in lines:
            if "Driver Version" in line:
                driver_version = line.split()[2]  
        self.log(f"驱动版本: {driver_version}")
                
        try:
            result = subprocess.run(
                ['nvcc','-V'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='GBK',
                text=True,
                check=True
            )
        except Exception as e:
            self.log(f"Failed to run nvcc: {e}")
            return False
        lines = result.stdout.split("\n")
        for line in lines:
            if 'cuda_' in line:
                cuda_version = line[11:15] 
        self.log(f"CUDA 版本: {cuda_version}")                
        
        try:
            result = subprocess.run(
                ['nvidia-smi','--query-gpu=name', '--format=csv'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='GBK',
                text=True,
                check=True
            )
        except Exception as e:
            self.log(f"Failed to run GPU 型号: {e}")
            return False
        lines = result.stdout.split("\n")
        
        for line in lines:
            if "NVIDIA" in line and "Tesla" in line or "GeForce" in line:
                gpu_model = line
        self.log(f"GPU 型号: {gpu_model}")
        self.log(f"GPU可以正常使用")
        return True
        
    @staticmethod
    def executableName(name):
        return name + ".exe" if os.name == "nt" else name

    def setDefaultParameters(self, parameterNode):
        if not parameterNode.GetParameter("Task"):
            parameterNode.SetParameter("Task", "骨骼")

    def logProcessOutput(self, proc, returnOutput=False):
        output = ""
        from subprocess import CalledProcessError
        while True:
            try:
                line = proc.stdout.readline()
                if not line:
                    break
                if returnOutput:
                    output += line
                self.log(line.rstrip())
            except UnicodeDecodeError as e:
                pass

        proc.wait()
        retcode = proc.returncode
        if retcode != 0:
            raise CalledProcessError(retcode, proc.args, output=proc.stdout, stderr=proc.stderr)
        return output if returnOutput else None

    def processVolume(self, inputFile, inputVolume, outputSegmentationFile, use_gpu,AutoOrganCommand):
        volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volumeStorageNode.SetFileName(inputFile)
        volumeStorageNode.UseCompressionOn()
        volumeStorageNode.WriteData(inputVolume)
        volumeStorageNode.UnRegister(None)
        # model_path = Path(__file__).parent / "AutoOrgan_bone_3mm.onnx"
        # options = ["-i", inputFile, "-o", outputSegmentationFile,"-m",model_path]
        
        if self.model_path == "":
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Warning", "请先选择模型路径。")
        
        self.log('开始分割，请稍等...')
        entry_main(inputFile,outputSegmentationFile,self.model_path,self.logCallback,use_gpu=False)
        # proc = slicer.util.launchConsoleProcess(AutoOrganCommand + options)
        # self.logProcessOutput(proc)
        return outputSegmentationFile

    def process(self,inputVolume,use_gpu) -> None:

        if not inputVolume:
            raise ValueError("没有输入图像，请选择输入图像")

        if self.model_path == "":
            qt.QMessageBox.warning(slicer.util.mainWindow(), "Warning", "请先选择模型路径。")
            raise ValueError("")

        startTime = time.time()
        tempFolder = slicer.util.tempDirectory()
        
        inputFile = tempFolder+"/input.nii"
        outputSegmentationFolder = tempFolder + "/segmentation.nii.gz"
        
        import sysconfig
        AutoOrganSlicerExecutablePath = os.path.join(sysconfig.get_path('scripts'), AutoOrganSlicerLogic.executableName("AutoOrgan"))
        pythonSlicerExecutablePath = shutil.which('PythonSlicer')
        
        if not pythonSlicerExecutablePath:
            raise RuntimeError("未发现Python环境")
        
        AutoOrganCommand = [ pythonSlicerExecutablePath, AutoOrganSlicerExecutablePath]
        outputSegmentationFile = self.processVolume(inputFile, inputVolume,outputSegmentationFolder, False, AutoOrganCommand)

        stopTime = time.time()
        self.log(f"\n分割完成,共耗时 {stopTime-startTime:.2f} 秒")

        # if self.clearOutputFolder:
        #     self.log("Cleaning up temporary folder...")
        #     if os.path.isdir(tempFolder):
        #         shutil.rmtree(tempFolder)
        # else:
        #     self.log(f"Not cleaning up temporary folder: {tempFolder}")
            
        return outputSegmentationFile,tempFolder
    
    def resample(self,inputVolume, is_label=False):
        tempFolder = slicer.util.tempDirectory()
        save_path = tempFolder+"/input.nii.gz"
        
        volumeStorageNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLVolumeArchetypeStorageNode")
        volumeStorageNode.SetFileName(save_path)
        volumeStorageNode.UseCompressionOn()
        volumeStorageNode.WriteData(inputVolume)
        volumeStorageNode.UnRegister(None)
        
        itk_image = sitk.ReadImage(save_path)
        itk_array = sitk.GetArrayFromImage(itk_image)
        itk_array = np.squeeze(itk_array)
        # self.log(itk_array.shape)
        itk_new_image = sitk.GetImageFromArray(itk_array)
        
        itk_new_image.SetDirection(itk_image.GetDirection())
        itk_new_image.SetOrigin(itk_image.GetOrigin())
        itk_new_image.SetSpacing(itk_image.GetSpacing())
        
        out_spacing=(2,2,2)
        original_spacing = itk_new_image.GetSpacing()
        original_size = itk_image.GetSize()
        out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                    int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                    int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(0)
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkBSpline)

        result_image = resample.Execute(itk_image)
        
        save_path = tempFolder+"/resample_input.nii.gz"
        sitk.WriteImage(result_image, r"C:\Users\27321\Desktop\ada.nii.gz")
        
        return save_path,tempFolder
    
class AutoOrganSlicerTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_AutoOrganSlicer1()
        self.setUp()
        self.test_AutoOrganSlicerSubset()

    def test_AutoOrganSlicer1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        inputVolume = SampleData.downloadSample('CTACardio')
        self.delayDisplay('Loaded test data set')

        outputSegmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')

        # Test the module logic

        # Logic testing is disabled by default to not overload automatic build machines (pytorch is a huge package and computation
        # on CPU takes 5-10 minutes). Set testLogic to True to enable testing.
        testLogic = False

        if testLogic:
            logic = AutoOrganSlicerLogic()
            logic.logCallback = self._mylog

            self.delayDisplay('Set up required Python packages')
            logic.setupPythonRequirements()

            self.delayDisplay('Compute output')
            logic.process(inputVolume, outputSegmentation, fast=False)

        else:
            logging.warning("test_AutoOrganSlicer1 logic testing was skipped")

        self.delayDisplay('Test passed')

    def _mylog(self,text):
        print(text)

    def test_AutoOrganSlicerSubset(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        inputVolume = SampleData.downloadSample('CTACardio')
        self.delayDisplay('Loaded test data set')

        outputSegmentation = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode')

        # Test the module logic

        # Logic testing is disabled by default to not overload automatic build machines (pytorch is a huge package and computation
        # on CPU takes 5-10 minutes). Set testLogic to True to enable testing.
        testLogic = False

        if testLogic:
            logic = AutoOrganSlicerLogic()
            logic.logCallback = self._mylog

            self.delayDisplay('Set up required Python packages')
            logic.setupPythonRequirements()

            self.delayDisplay('Compute output')
            _subset = ["lung_upper_lobe_left","lung_lower_lobe_right","trachea"]
            logic.process(inputVolume, outputSegmentation, fast = False, subset = _subset)

        else:
            logging.warning("test_AutoOrganSlicer1 logic testing was skipped")

        self.delayDisplay('Test passed')
