import random
import cupy as cp
import SimpleITK as sitk
from cucim.skimage import measure
from skimage.segmentation import flood

def get_keys_by_value(dictionary, value):
    keys = []
    for key, val in dictionary.items():
        if val == value:
            keys.append(key)
    return keys[0]

def clear_small_region(array,rate = 0.01):
    connect_array,connect_number = measure.label(array,return_num=True,connectivity=1)
                    
    areas = []
    points = []
    for region in measure.regionprops(connect_array):
        areas.append(region.area)
        points.append(region.coords)        
    
    areas = cp.array(areas)
    max_area = areas[cp.argmax(areas)] * rate
    indexs = cp.where(areas < max_area)
    
    for item in indexs[0]:
        for index in points[int(item)]:
            array[index[0], index[1],index[2]] = 0    
        
    return array

def get_max_region(array,number = 1):
    connect_array = measure.label(array,connectivity=1)
    
    areas = []
    points = []
    for region in measure.regionprops(connect_array):
        areas.append(region.area)
        points.append(region.coords)    
    
    areas = cp.array(areas)
    
    if areas.size < number:
        return array
    
    sorted_indices = cp.argsort(areas)[:-number]
            
    for item in sorted_indices:
        index = int(item)
        points_array = points[index]
        for array_item in points_array:
            array[array_item[0],array_item[1],array_item[2]] = 0
    return array

def keep_setting(array,target_image):
    result_image = sitk.GetImageFromArray(array)
    result_image.SetDirection(target_image.GetDirection())
    result_image.SetOrigin(target_image.GetOrigin())
    result_image.SetSpacing(target_image.GetSpacing())
    return result_image

def get_max_region_central_location(array):
    connect_array = measure.label(array,connectivity=1)
    
    areas = []
    points = []
    for region in measure.regionprops(connect_array):
        areas.append(region.area)
        points.append(region.centroid)
    
    areas = cp.array(areas)
    sorted_indices = cp.argmax(areas)
    return points[int(sorted_indices)]
    
def get_max_sitk(in_img):
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    output_mask = cc_filter.Execute(in_img)
 
    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)
 
    num_connected_label = cc_filter.GetObjectCount()
 
    area_max_label = 0
    area_max = 0
 
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  
        if area > area_max:
            area_max_label = i
            area_max = area
 
    np_output_mask = sitk.GetArrayFromImage(output_mask)
 
    res_mask = cp.zeros_like(np_output_mask)
    res_mask[np_output_mask == area_max_label] = 1
 
    res_itk = keep_setting(res_mask,in_img)
    return res_itk

def remove_label(image_array,label):
    flag = False
    
    if label in image_array:
        flag = True
        label_loc = image_array == label
        image_array[label_loc] = 0
        tem_arr = cp.zeros_like(image_array)
        tem_arr[label_loc] = 1
        label_loc =  get_max_region(tem_arr) != 0
        return flag,label_loc
    return flag,0

def fix_direction(image_array,json_data):
    
    labels =  list(json_data['labels'].keys())
    
    bone_left = []
    bone_right = []
    for item in labels:
        if 'left' in item:
            bone_left.append(json_data['labels'][item])
        elif 'right' in item:
            bone_right.append(json_data['labels'][item]) 
            
    left_array = image_array[:,:,int(image_array.shape[2] / 2):]
    
    for i in range(len(bone_left)):
        image_array[image_array == bone_left[i]] = bone_right[i]
        left_array[left_array == bone_right[i]] = bone_left[i]
    
    return image_array

def fix_rib(image_array,json_data):
    labels =  list(json_data['labels'].keys())
    rib_right_12 = json_data['labels']['rib_right_12']
    rib_left_12 = json_data['labels']['rib_left_12']
    
    location = image_array == json_data['labels']['vertebrae_T1']
    if cp.count_nonzero(location) <= 0:
        return image_array
    
    rib_left_label = []
    rib_right_label = []
    for item in labels:
        if 'rib_left' in item:
            rib_left_label.append(json_data['labels'][item])
        elif 'rib_right' in item:
            rib_right_label.append(json_data['labels'][item])
    
    rib_number = 24
    
    location = image_array == rib_right_12
    if cp.count_nonzero(location) == 0:
        rib_number -= 1
        rib_left_label.remove(rib_left_12)
        
    location = image_array == rib_left_12
    if cp.count_nonzero(location) == 0:
        rib_number -= 1
        rib_right_label.remove(rib_right_12)
    
    all_labels_array = cp.array(rib_right_label + rib_right_label)
    new_array = cp.zeros_like(image_array)
    rib_mask = cp.isin(image_array,all_labels_array)
    new_array[rib_mask] = 1
    
    new_array = clear_small_region(new_array)
    connect_array,connect_number = measure.label(new_array,return_num=True,connectivity=1)
    
    if connect_number < rib_number:
        return image_array
    
    image_array[rib_mask] = 0
    
    areas = []
    points = []
    for region in measure.regionprops(connect_array):
        areas.append(region.area)
        points.append(region.coords)
    
    areas = cp.array(areas)
    sorted_indices = cp.argsort(areas)[:-rib_number]
    for item in sorted_indices:
        index = int(item)
        array = points[index]
        for array_item in array:
            connect_array[array_item[0],array_item[1],array_item[2]] = 0
    
    label_dict = {}
    _label_dict = {}
    for region in measure.regionprops(connect_array):
        label_dict[region.label] = region.centroid
        _label_dict[region.label] = max(cp.where(connect_array == region.label)[0])
        
    vertebra_array = cp.zeros_like(image_array)
    vertebra_label = json_data['labels']['vertebrae_T1']
    vertebra_array[image_array == vertebra_label] = vertebra_label
    vertebra_centre_location = get_max_region_central_location(vertebra_array)        
        
    right_label_location = {}
    left_label_location = {}    
    for label,centroid in zip(label_dict.keys(),label_dict.values()):
        if centroid[2] < vertebra_centre_location[2]:
            right_label_location[label] = _label_dict[label]
        else:
            left_label_location[label] = _label_dict[label]

    _right_label_loc = [i[0] for i in sorted(right_label_location.items(), key=lambda x: (x[1],x[0]),reverse=True)]
    _left_label_loc = [i[0] for i in sorted(left_label_location.items(), key=lambda x: (x[1],x[0]),reverse=True)]
    
    all_labels = _right_label_loc + _left_label_loc
    true_labels = rib_right_label + rib_left_label

    for i in range(len(true_labels)):
        image_array[connect_array == all_labels[i]] = true_labels[i]

    return image_array
    
def fix_vertebra(image_array,json_data):
    
    vertebra_list = []
    labels_list = list(json_data['labels'].keys())
    
    for item in labels_list:
        if 'vertebrae' in item:
            vertebra_list.append(json_data['labels'][item])
            
    vertebra_array = cp.zeros_like(image_array)
    for label in vertebra_list:
        vertebra_array[image_array == label] = label
    
    true_label = list(cp.unique(vertebra_array))
    true_label.sort()
    true_label.remove(0)
    
    for label in range(len(true_label)):
        temporary_array = cp.zeros_like(image_array)
        temporary_array[vertebra_array == true_label[label]] = true_label[label]
        connect_array,connect_number = measure.label(temporary_array,return_num=True)
        
        if connect_number == 1:
            continue
        
        elif connect_number > 1:
            
            areas = []
            labels = []
        
            for item in measure.regionprops(connect_array):
                areas.append(item.area)
                labels.append(item.label)
            
            max_areas = max(areas)
            
            for index in range(len(areas)):
                if areas[index] == max_areas:
                    del labels[index]
                    break            
            
            if (label == 0) or (label == len(true_label) - 1):
                if label == 0 and len(true_label) > 1:
                    other_label = int(true_label[1])
                else:
                    other_label = int(true_label[label - 1])
                
                other_array = cp.zeros_like(image_array)
                other_array[temporary_array == other_label] = other_label
                _,other_connect_number = measure.label(other_array,return_num=True)
                
                for label_item in labels:
                    temporary_location = connect_array == label_item
                    other_array[temporary_location] = other_label
                    _,other_merged_connect_number = measure.label(other_array,return_num=True)
                    
                    if other_connect_number >= other_merged_connect_number:
                        image_array[temporary_location] = other_label
                    
                    other_array[temporary_location] = 0
                    image_array[temporary_location] = 0
            else:
                up = int(true_label[label - 1])
                down = int(true_label[label + 1])
                
                up_array = cp.zeros_like(vertebra_array)
                up_array[vertebra_array == up] = up
                
                down_array = cp.zeros_like(vertebra_array)
                down_array[vertebra_array == down] = down
                
                _,up_before_number = measure.label(up_array,return_num=True)
                _,down_before_number = measure.label(down_array,return_num=True)
                
                for label_item in labels:
                    temporary_location = connect_array == label_item
                    
                    up_array[temporary_location] = up
                    _,up_after_number = measure.label(up_array,return_num=True)
                    
                    if up_before_number >= up_after_number:
                        image_array[temporary_location] = up 
                        up_array[temporary_location] = 0
                        continue
                                           
                    down_array[temporary_location] = down
                    _,down_after_number = measure.label(down_array,return_num=True)
                    
                    if down_before_number >= down_after_number:
                        image_array[temporary_location] = down          
                        down_array[temporary_location] = 0
                        continue
                    
                    up_array[temporary_location] = 0
                    down_array[temporary_location] = 0
                    image_array[temporary_location] = 0
                    
    return image_array

def fix_limb(image_array,json_data):
    import numpy as cp
    from skimage import measure
    
    left = json_data['labels']['left_scapula']
    right = json_data['labels']['right_scapula']
    limb_label = json_data['labels']['humerus____ulna____radius____carpal____metacarpal_bone____phalanx']
    
    temporary_array = cp.copy(image_array)
    temporary_array[(image_array == left) | (image_array == 0) | (image_array == right)] = 250
    
    limb_array = cp.zeros_like(image_array)
    limb_array[image_array == limb_label] = limb_label
    
    limb_connect_array = measure.label(limb_array)
    
    properties = measure.regionprops(limb_connect_array)
    for prop in properties:
        location_list = []
        label_location = cp.where(limb_connect_array == prop.label)
        random_index = random.randint(0,len(label_location[0]))
        
        for i in range(0,len(label_location)):
            location_list.append(label_location[i][random_index-1])
            
        mask = flood(temporary_array,tuple(location_list),tolerance=100)
        image_array[mask] = limb_label
    return image_array

def fix_noisy(image_array,json_data):

    labels = list(json_data['labels'].values())
    labels.remove(0)
    labels.remove(json_data['labels']['humerus____ulna____radius____carpal____metacarpal_bone____phalanx'])
    labels.remove(json_data['labels']['right_toe_bone'])
    labels.remove(json_data['labels']['left_toe_bone'])
    
    for label in labels:
        new_array = cp.zeros_like(image_array)
        
        location = image_array == label
        image_array[location] = 0
        new_array[location] = label
        
        new_array = get_max_region(new_array)
        new_location = new_array != 0
        image_array[new_location] = label
        
    return image_array
    
def work(image_array,json_data):
    
    # transform numpy->cupy
    image_array = cp.array(image_array)
    
    # postprocess 
    image_array = fix_direction(image_array,json_data)
    image_array = fix_rib(image_array,json_data)
    image_array = fix_vertebra(image_array,json_data)
    
    # tranform cupy->numpy
    return image_array.get()
    # image_array = fix_noisy(image_array,json_data)
    # image_array = fix_limb(image_array,json_data)