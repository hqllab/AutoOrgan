import numpy as np
from munch import DefaultMunch
from functools import lru_cache
from skimage.transform import resize
from scipy.ndimage import map_coordinates,gaussian_filter

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
def compute_gaussian(tile_size, sigma_scale,value_scaling_factor, dtype=np.float16 ) :
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

def preprocess(data,original_spacing,plans_dict):
    parameters_dict = DefaultMunch()
    parameters_dict.origin_shape = data.shape
    
    cropped_data,crop_bbox,crop_list = crop_to_nonzero(data) 
    parameters_dict.shape_after_crop = cropped_data.shape
    parameters_dict.crop_bbox = crop_bbox
    parameters_dict.crop_list = crop_list
    
    cropped_normed_data = ct_znorm(cropped_data,plans_dict.foreground_intensity_properties_per_channel)
    
    target_spacing = plans_dict.original_median_spacing_after_transp
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
    
    return cropped_normed_resampled_patched_data,slicers,gaussian,parameters_dict