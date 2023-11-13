import pandas as pd
import json
from . import create_scenarios
import os

def create_files(dc, data_dr, perspective=False, set='train'):
        if perspective == False:
            folder = os.path.join(data_dr,'non_perspective')
        else:
            folder = os.path.join(data_dr, 'perspective')   
        
        if not os.path.exists(folder): os.mkdir(folder)
        
        for quality, dic in dc.items():
             #create folder for the entire quality scenario
             quality_folder = os.path.join(folder,quality)
             if not os.path.exists(quality_folder): os.mkdir(quality_folder)                             
             for scenario, data_ls in dic.items():
                  scenario_path = os.path.join(quality_folder,f'{scenario}_{set}.json')
                  with open(scenario_path, 'w') as outfile:
                       data_obj = json.dumps(data_ls, indent=4)
                       outfile.write(data_obj)


def main(train_p, dev_p, data_dir):
    #NON PERSPECTIVE
    #extract scenarios
    non_persp_train_dc = create_scenarios.main(train_p, perspective=False)
    non_persp_dev_dc = create_scenarios.main(dev_p, perspective=False)
    #store_files
    create_files(non_persp_train_dc, data_dir, perspective=False, set='train')
    create_files(non_persp_dev_dc, data_dir, perspective=False, set='dev')

    #PERSPECTIVE
    #extract scenarios
    persp_train_dc = create_scenarios.main(train_p, perspective=True)
    persp_dev_dc = create_scenarios.main(dev_p, perspective=True)
    #store_files
    create_files(persp_train_dc, data_dir, perspective=True, set='train')
    create_files(persp_dev_dc, data_dir, perspective=True, set='dev')




if __name__ == "__main__":
    print('Running create_files.py')
    
    