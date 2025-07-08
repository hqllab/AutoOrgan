import onnx
import torch
import argparse
import onnxruntime
import numpy as np
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import nnunetv2
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

# 两种方法都可以使用
def export_onnx_model(model_path,output_path,batch_size,folds='0',test_accuracy=True,verbose: bool = False):

    use_dynamic_axes = batch_size == 0
    
    if output_path is None:
        output_path = Path(__file__).parent / f"nnunetv2_model.onnx"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
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

def _export_onnx_model(input_path,output_path,batch_size,folds='0',test_accuracy=False,verbose: bool = False):
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = Path(__file__).parent / f"nnunetv2_model.onnx"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    dataset_json = load_json(join(input_path, 'dataset.json'))
    plans = load_json(join(input_path, 'plans.json'))
    plans_manager = PlansManager(plans)
    parameters = []

    checkpoint_name = 'checkpoint_final.pth'

    checkpoint = torch.load(input_path / f'fold_{folds}' / checkpoint_name,map_location=torch.device('cpu'))
    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']

    parameters.append(checkpoint['network_weights'])
    configuration_manager = plans_manager.get_configuration(configuration_name)
    # restore network
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
    trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                trainer_name, 'nnunetv2.training.nnUNetTrainer')

    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False
    )
    network.load_state_dict(parameters[0])
    network.eval()
    
    patch_size = configuration_manager.patch_size
    use_dynamic_axes = batch_size == 0
    if use_dynamic_axes:
        rand_input = torch.rand((1, 1, * patch_size))
        torch.onnx.export(network,rand_input,output_path,export_params=True,verbose=verbose,input_names=["input"],output_names=["output"],dynamic_axes={"input": {0: "batch_size"},"output": {0: "batch_size"}})
    else:
        rand_input = torch.rand((batch_size, 1, * patch_size))
        torch.onnx.export(network,rand_input,output_path,export_params=True,verbose=verbose,input_names=["input"],output_names=["output"])
    
    if test_accuracy:
        print("Testing accuracy...")
        torch_output = network(rand_input)
        
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

    parser.add_argument('-input_path', type=str,help='', default='/home/wangnannan/nnunet_dir/nnUNet_results/Dataset106_TotalBone/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres')
    parser.add_argument('-output_path', type=str,default='/home/wangnannan/nnunet_dir/nnUNet_results/Dataset106_TotalBone/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/model.onnx')
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-folds', type=str, default='0')

    args = parser.parse_args()
     
    export_onnx_model(args.input_path,args.output_path,args.batch_size,args.folds)
