import argparse
import SimpleITK as sitk
from munch import DefaultMunch

from AutoOrgan.config import config_dict,labels_dict
from AutoOrgan.backend_data import BackendImage
from AutoOrgan.preprocess import preprocess
from AutoOrgan.infer import run_infer_ONNX
from AutoOrgan.postprocess import postprocess
from AutoOrgan.fixed import work

def infer(receive_dict,config_dict,model_path,use_gpu):
    receive_dict = DefaultMunch.fromDict(receive_dict)
    
    ct_data = receive_dict.InImage.CTImage.Data
    ct_meta = receive_dict.InImage.CTImage.Meta 
    ct_image = BackendImage(ct_data, ct_meta)
    ct_image.flag = 'ct'
        
    classes_count = 73
    ct_image,slicers,gaussian,ct_parameters_dict = preprocess(ct_image, config_dict)
    ct_image = run_infer_ONNX(ct_image,slicers,gaussian,classes_count,model_path,use_gpu)
    
    ct_image = postprocess(ct_image,ct_parameters_dict)
    ct_image = work(ct_image,labels_dict)
    
    return ct_image

def entry_main(input_path,output_path,model_path,use_gpu):
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
    
    infer_result = infer(parameters_dict,config_dict,model_path,use_gpu)
    infer_array = infer_result.infer_data
    image = sitk.GetImageFromArray(infer_array)
    image.SetSpacing((ct_x_spacing,ct_y_spacing,ct_z_spacing))
    image.SetOrigin(ct_image.GetOrigin())
    image.SetDirection(ct_image.GetDirection())
    sitk.WriteImage(image,output_path)

def main_entry():
    parser = argparse.ArgumentParser(description="AutoOrgan")

    parser.add_argument('-i', type=str, required=True, help='input path')
    parser.add_argument('-o', type=str,  help='out path')
    parser.add_argument('-m', type=str,  help='model path')
    parser.add_argument('-g', action='store_true', help='use gpu')

    args = parser.parse_args()
    entry_main(args.i,args.o,args.m,args.g)
    
# entry_main(r"C:\Users\27321\Desktop\Patient_0048.nii.gz",r"C:\Users\27321\Desktop\out.nii.gz",r"C:\Users\27321\Desktop\checkpoint_final.onnx",True)