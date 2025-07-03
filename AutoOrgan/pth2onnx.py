import onnx
import torch
import argparse
import onnxruntime
import numpy as np
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.file_path_utilities import get_output_folder

def export_onnx_model(model_path,output_path,batch_size,folds='0',test_accuracy=False,verbose: bool = False):

    use_dynamic_axes = batch_size == 0
    
    if output_path is None:
        output_path = Path(__file__).parent / f"nnunetv2_model.onnx"
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = 'checkpoint_final.pth'
    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(model_training_output_dir=model_path,use_folds=folds,checkpoint_name=checkpoint_name)
    
    list_of_parameters = predictor.list_of_parameters
    network = predictor.network
    config = predictor.configuration_manager
    
    network.load_state_dict(list_of_parameters[0])
    network.eval()

    if use_dynamic_axes:
        rand_input = torch.rand((1, 1, *config.patch_size))
        torch_output = network(rand_input)
        torch.onnx.export(network,rand_input,output_path,export_params=True,verbose=verbose,input_names=["input"],output_names=["output"],dynamic_axes={"input": {0: "batch_size"},"output": {0: "batch_size"}})
    else:
        rand_input = torch.rand((batch_size, 1, *config.patch_size))
        torch_output = network(rand_input)
        torch.onnx.export(network,rand_input,output_path,export_params=True,verbose=verbose,input_names=["input"],output_names=["output"])

    if test_accuracy:
        print("Testing accuracy...")
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_inputs = {ort_session.get_inputs()[0].name: rand_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        try:
            np.testing.assert_allclose(
                torch_output.detach().cpu().numpy(),
                ort_outs[0],
                rtol=1e-03,
                atol=1e-05,
                verbose=True,
            )
        except AssertionError as e:
            print("WARN: Differences found between torch and onnx:\n")
            print(e)
            print(
                "\nExport will continue, but please verify that your pipeline matches the original."
            )

    print(f"Exported {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nnUnetv2 pth model to onnx")

    parser.add_argument('-input_path', type=str,help='', default='/home/wangnannan/nnunet_dir/nnUNet_results/Dataset007_Bone/nnUNetTrainer__nnUNetPlans__3d_fullres')
    parser.add_argument('-output_path', type=str)
    parser.add_argument('-batch_size', type=int, default=0)
    parser.add_argument('-folds', type=str, default='0')

    args = parser.parse_args()
     
    export_onnx_model(args.input_path,args.output_path,args.batch_size,args.folds)
