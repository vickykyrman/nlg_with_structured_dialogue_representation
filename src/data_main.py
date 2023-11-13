import os
import args
from data_utils import rearrange
from data_utils import extract_history
from data_utils import create_files
from data_utils.perspective_utils import extract_perspective

def main():
    #we call the .ARGS attribute of the class object args. 
    ARGS = args.ARGS
    classifier_path = ARGS.dial_act_classifier

    #REARRANGE DATA
    original_data_path =  ARGS.original_path
    rearranged_data_path = os.path.join(os.path.dirname(original_data_path),'rearranged.csv')

    print('Rearranging original data...')
    print()
    rearrange.main(original_data_path, rearranged_data_path)
    print(f'Rearranged data stored in {os.path.abspath(rearranged_data_path)}')
    print()

    #CREATE HISTORY
    history_path = os.path.join(os.path.dirname(original_data_path),'history.csv')
    train_path = os.path.join(os.path.dirname(original_data_path),'train.csv')
    dev_path = os.path.join(os.path.dirname(original_data_path),'dev.csv')

    print('Extracting dialogue history...')
    print()
    extract_history.main(rearranged_data_path, history_path, train_path, dev_path, classifier_path)
    print(f'Dialogue history diata stored in {os.path.abspath(history_path)}')
    print()
    print(f'Train data stored in {os.path.abspath(train_path)}')
    print()
    print(f'Test data stored in {os.path.abspath(dev_path)}')
    print()

    #CREATE INPUT SCENARIOS AND STORE DATA
    print('Creating input scenarios and storing files...')
    print()     
    data_directory = os.path.dirname(original_data_path)
    create_files.main(train_path, dev_path, data_directory)
    print("YOU ARE ALL SET!")

if __name__=='__main__':
    main()