import json
import pprint

def convert_json_to_py(json_file, py_file, var_name='config'):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(py_file, 'w', encoding='utf-8') as f:
        # f.write(f"{var_name} = {repr(data)}\n")
        f.write(f"{var_name} = ")
        pprint.pprint(data, stream=f)
    print(f"成功生成 {py_file}")

# 示例调用
if __name__ == "__main__":
    convert_json_to_py('/home/wangnannan/nnunet_dir/nnUNet_results/Dataset998_TotalRib/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/dataset.json', 'dataset.py', 'config')