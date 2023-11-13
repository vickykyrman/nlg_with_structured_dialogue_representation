import os
import args
from model_utils import train, test



def main():
    #we call the .ARGS attribute of the class object args. 
    ARGS = args.ARGS

    models_directory = ARGS.models_path
    original_data_path =  ARGS.mini_original_path
    data_directory =  os.path.dirname(original_data_path)

    quality = ARGS.quality
    quantity = ARGS.quantity
    perspective_info = ARGS.perspective

    if not os.path.exists(models_directory): os.mkdir(models_directory)

    if ARGS.mode == 'train':
        train.main(data_directory, models_directory, quality, quantity, ARGS, perspective = perspective_info)
    
    
    elif ARGS.mode == 'evaluate': 
        #create evaluation folders
        evaluation_directory = ARGS.evaluation_path
        if not os.path.exists(evaluation_directory): os.mkdir(evaluation_directory)
        automatic_scores_directory = os.path.join(evaluation_directory,'automatic_scores')
        if not os.path.exists(automatic_scores_directory): os.mkdir(automatic_scores_directory)

        test.main(data_directory, models_directory, quality, quantity, ARGS, automatic_scores_directory, perspective=perspective_info)





    


   


if __name__=='__main__':
    main()
        


