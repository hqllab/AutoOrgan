import numpy as np
from skimage.transform import resize

def postprocess(image, parameters_dict):
    
    image_data = image.infer_data
    
    slicer_revert_padding = parameters_dict.pad_bbox
    crop_image_data = image_data[tuple([slice(None), *slicer_revert_padding])]
    crop_image_data = np.argmax(crop_image_data,0)
    
    # if image.flag == 'ct':
    #     image.infer_data = crop_image_data.astype(np.uint16)
    #     return image
    
    crop_zoom_image_data = resize(crop_image_data,parameters_dict.shape_after_crop,order=0)
    
    slicer = tuple([slice(*i) for i in parameters_dict.crop_bbox])
    segmentation_reverted_cropping = np.zeros(parameters_dict.origin_shape,dtype=np.uint16)
    segmentation_reverted_cropping[slicer] = crop_zoom_image_data
    image_data = segmentation_reverted_cropping
    
    image.infer_data = image_data.astype(np.uint16)
    return image