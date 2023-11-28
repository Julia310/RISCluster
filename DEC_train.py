from RISCluster.production import train
from RISCluster import utils
import os

# Main project folder to save outputs:
project_folder = '.'
# Path to configuration files:
path_config = f"{project_folder}/Config"
# Path to folder containing data, including HDF file for ML workflow:
path_data = f"{project_folder}/Data"
# Path to raw seismic data:
# path_data_seismo = f"{path_data}/Seismo"
path_data_seismo = "./zdata2/data/wfjenkin/RIS_Seismic"
# Path to save workflow outputs (ML models, figures, results, etc.)
path_output = f"{project_folder}/Outputs"
# Path to HDF dataset:
fname_dataset = f"{path_data}/RISData_20210713.h5"
# Path to save paper-ready figures:
figure_savepath = f"{path_output}/Figures"

exp_name = "FullArray"
universal = {
    'exp_name': exp_name,
    'fname_dataset': fname_dataset,
    'savepath': path_output,
    'indexpath': os.path.join(path_data, 'TraValIndex_M=50000.pkl'),
    'configpath': path_config
}
device_no = 1

batch_size = 64
LR = 0.001

expserial = 'Exp20210727T192309'
runserial = f'Run_BatchSz={batch_size}_LR={LR}'
# exp_path = f"{path_output}/Models/AEC/{expserial}/{runserial}"
exp_path_AEC = os.path.join(path_output, 'Models', 'AEC', expserial, runserial)

weights_AEC = os.path.join(exp_path_AEC, 'AEC_Params_Final.pt')
print(weights_AEC)



parameters = {
    'model': 'DEC',
    'mode': 'train',
    'n_epochs': 400,
    'show': False,
    'send_message': False,
    'transform': 'vec_norm',
    'tb': True,
    'tbport': 6999,
    'workers': 4,
    'loadmode': 'ram',
    'datafiletype': 'h5',
    'init': 'load',
    'update_interval': -1,
    'saved_weights': weights_AEC
}

hyperparameters = {
    'batch_size': '64',
    'lr': '0.001',
    'n_clusters': '10',
    'gamma': '0.001',
    'tol': 0.003
}
init_path = utils.config_training(universal, parameters, hyperparameters)
config_DEC = utils.Configuration(init_path)
config_DEC.load_config()
#config_DEC.set_device(device_no)
config_DEC.show = True