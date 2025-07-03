import random
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage import measure
from skimage.segmentation import flood

def keep_setting(array,target_image):
    result_image = sitk.GetImageFromArray(array)
    result_image.SetDirection(target_image.GetDirection())
    result_image.SetOrigin(target_image.GetOrigin())
    result_image.SetSpacing(target_image.GetSpacing())
    return result_image

def clear_small_region(array,rate = 0.02):
    connect_array,_connect_number = ndimage.label(array)
                    
    sizes  = np.bincount(connect_array.ravel())
    
    min_volume_count = np.max(sizes[1:])  * rate
    small_components = np.where(sizes < min_volume_count)[0]
    
    array[np.isin(connect_array,small_components)] = 0
    
    return array

def get_max_region(array):
    connect_array,_ = ndimage.label(array)
    sizes  = np.bincount(connect_array.ravel())
    max_label = np.argmax(sizes[1:]) + 1
    
    result = np.zeros_like(array)
    location = connect_array == max_label
    result[location] = array[location]
    return result

def get_max_region_central_location(array):
    # 输入为一个numpy数组，返回数组中最大连通域的中心坐标，返回类型为tuple
    connect_array = measure.label(array,connectivity=1)
    areas = []
    points = []
    for region in measure.regionprops(connect_array):
        areas.append(region.area)
        points.append(region.centroid)
    
    areas = np.array(areas)
    sorted_indices = np.argmax(areas)
    return points[int(sorted_indices)]
    
def get_max_SimpleITK(itk_image):
    # 输入为一个Simpleitk.Image，只保留图像的最大连通域，其他连通域置零。返回类型为Simpleitk.Image
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(False)
    output_mask = cc_filter.Execute(itk_image)
 
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
 
    result_mask = np.zeros_like(np_output_mask)
    result_mask[np_output_mask == area_max_label] = 1
 
    result_itk_image = keep_setting(result_mask,itk_image)
    return result_itk_image

def fix_direction(image_array,json_data):
    
    labels = list(json_data['labels'].keys())
    
    bone_left = []
    bone_right = []
    
    for item in labels:
        if 'left' in item: 
            bone_left.append(json_data['labels'][item])
        elif 'right' in item: 
            bone_right.append(json_data['labels'][item]) 
            
    size = int(image_array.shape[2] / 2)
    
    for i in range(len(bone_left)):
        image_array[image_array == bone_left[i]] = bone_right[i]
        left = image_array[:,:,size:]
        left[left == bone_right[i]] = bone_left[i]
    
    return image_array

def fix_rib(image_array,json_data):
    labels =  list(json_data['labels'].keys())
    rib_right_12 = json_data['labels']['rib_right_12']
    rib_left_12 = json_data['labels']['rib_left_12']
    
    location = image_array == json_data['labels']['vertebrae_T1']
    if np.count_nonzero(location) <= 0:
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
    if np.count_nonzero(location) == 0:
        rib_number -= 1
        rib_left_label.remove(rib_left_12)
        
    location = image_array == rib_left_12
    if np.count_nonzero(location) == 0:
        rib_number -= 1
        rib_right_label.remove(rib_right_12)
    
        
    all_labels_array = np.array(rib_right_label + rib_left_label)
    new_array = np.zeros_like(image_array)
    rib_mask = np.isin(image_array,all_labels_array)
    new_array[rib_mask] = 1
    new_array = clear_small_region(new_array)
    connect_array,connect_number = measure.label(new_array,return_num=True,connectivity=1)
    
    if connect_number == 22:
        rib_number = 22
        rib_left_label.remove(rib_left_12)
        rib_right_label.remove(rib_right_12)
    
    if connect_number < rib_number:
        return None    
    
    image_array[rib_mask] = 0
    
    areas = []
    points = []
    for region in measure.regionprops(connect_array):
        areas.append(region.area)
        points.append(region.coords)
    
    areas = np.array(areas)
    sorted_indices = np.argsort(areas)[:-rib_number]
    for item in sorted_indices:
        index = int(item)
        array = points[index]
        for array_item in array:
            connect_array[array_item[0],array_item[1],array_item[2]] = 0
    
    label_dict = {}
    _label_dict = {}
    for region in measure.regionprops(connect_array):
        label_dict[region.label] = region.centroid
        _label_dict[region.label] = max(np.where(connect_array == region.label)[0])
        
    vertebra_array = np.zeros_like(image_array)
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
            
    vertebra_array = np.zeros_like(image_array)
    for label in vertebra_list:
        vertebra_array[image_array == label] = label
    
    true_label = list(np.unique(vertebra_array))
    true_label.sort()
    true_label.remove(0)
    
    for label in range(len(true_label)):
        temporary_array = np.zeros_like(image_array)
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
                
                other_array = np.zeros_like(image_array)
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
                
                up_array = np.zeros_like(vertebra_array)
                up_array[vertebra_array == up] = up
                
                down_array = np.zeros_like(vertebra_array)
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
                    
        # temporary_array = np.zeros_like(image_array)
        # temporary_array[vertebra_array == true_label[label]] = true_label[label]        
        # connect_array,connect_number = measure.label(temporary_array,return_num=True)    
              
        # if connect_number > 1:
        #     areas = [r.area for r in measure.regionprops(conn_arr)]
        #     areas.sort()
        #     labels = [r.label for r in measure.regionprops(conn_arr) if r.area != areas[-1]]
            
        #     if (index == 0) or (index == len(true_label) - 1):
        #         if index == 0 and len(true_label) > 1:
        #             other = int(true_label[1])
        #         else:
        #             other = int(true_label[index - 1])
                    
        #         other_loc = get_label_cent(zhui_arr,other)
        #         _loc = get_label_cent(zhui_arr,int(true_label[index]))
        #         for item in labels:
        #             loc = np.where(conn_arr == item)
        #             for l in zip(loc[0],loc[1],loc[2]):
        #                 other_num = (other_loc[2] - l[0])**2
        #                 _num = (_loc[2] - l[0])**2
        #                 if min(other_num,_num) == other_num:
        #                     img_arr[l] = other
            
        #     else:
        #         up = int(true_label[index - 1])
        #         down = int(true_label[index + 1])
                
        #         up_gra = get_label_cent(zhui_arr,up)
        #         down_gra = get_label_cent(zhui_arr,down)
        #         _gra = get_label_cent(zhui_arr,int(true_label[index]))
                
        #         for item in labels:
        #             loc = np.where(conn_arr == item)
        #             for l in  zip(loc[0],loc[1],loc[2]):
        #                 down_num = (down_gra[2] - l[0])**2 
        #                 up_num = (up_gra[2] - l[0])**2 
        #                 _num = (_gra[2] - l[0])**2
        #                 if  min(up_num,down_num,_num) == up_num :
        #                     img_arr[loc] = up
        #                 elif min(up_num,down_num,_num) == down_num:
        #                     img_arr[loc] = down
        #             up_num = ((up_gra[0] - loc[0])**2 + (up_gra[1] - loc[1])**2 + (up_gra[2] - loc[2])**2)
        #             down_num = ((down_gra[0] - loc[0])**2 + (down_gra[1] - loc[1])**2 + (down_gra[2] - loc[2])**2)
    
    return image_array

def fix_limb(image_array,json_data):
    left = json_data['labels']['left_scapula']
    right = json_data['labels']['right_scapula']
    limb_label = json_data['labels']['humerus____ulna____radius____carpal____metacarpal_bone____phalanx']
    
    temporary_array = np.copy(image_array)
    temporary_array[(image_array == left) | (image_array == 0) | (image_array == right)] = 250
    
    limb_array = np.zeros_like(image_array)
    limb_array[image_array == limb_label] = limb_label
    
    limb_connect_array = measure.label(limb_array)
    
    properties = measure.regionprops(limb_connect_array)
    for prop in properties:
        location_list = []
        label_location = np.where(limb_connect_array == prop.label)
        random_index = random.randint(0,len(label_location[0]))
            
        for i in range(0,len(label_location)):
            location_list.append(label_location[i][random_index-1])
            
        mask = (limb_connect_array == prop.label)
        indices = np.argwhere(mask)
        if indices.size > 0:
            random_index = indices[np.random.choice(indices.shape[0])]     
                   
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
        location = image_array == label
        
        if np.count_nonzero(location) <= 0:
            continue
        
        label_array = np.zeros_like(image_array)
        image_array[location] = 0
        label_array[location] = label
        
        image_array[get_max_region(label_array) != 0] = label
        
    return image_array

def work(image,json_data):
    # 对数据依次进行 修正方向、肋骨、椎骨、假阳性、上肢等操作。
    image_array = np.array(image.infer_data)
    # image_array = np.flip(np.array(image.infer_data),axis=0)
    
    image_array = fix_direction(image_array,json_data)
    image_array = fix_rib(image_array,json_data)
    # image_array = fix_vertebra(image_array,json_data)
    # image_array = fix_noisy(image_array,json_data)
    # image_array = fix_limb(image_array,json_data)
    
    image.infer_data = image_array
    
    return image