from munch import Munch

config_dict = Munch()

config_dict.device = 'cuda:0'
config_dict.use_cpus = 1

config_dict.input_folder = ''
config_dict.output_folder = ''
config_dict.model_folder = ''
config_dict.temporary_folder = ''