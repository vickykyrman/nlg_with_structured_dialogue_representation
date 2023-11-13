import json
import pandas as pd
import random
import numpy as np
import os

#set seed
np.random.seed(128)
random.seed(128)

def rescale(col):
    return col.apply(lambda x: x*100)

def rearrange(data):
    new_data = []
    #BLEU
    for dict in data:
        dict['BLEU']['1-gram-precision'] = dict['BLEU']['precisions'][0]
        dict['BLEU']['2-gram-precision'] = dict['BLEU']['precisions'][1]
        dict['BLEU']['3-gram-precision'] = dict['BLEU']['precisions'][2]
        dict['BLEU']['4-gram-precision'] = dict['BLEU']['precisions'][3] 
        del dict['BLEU']['precisions']
    
        #BERTSCORE
        del dict['BERTSCORE']['hashcode']

        #merge all dicts
        new_dict = {'model':dict['model']}
        for key in list(dict.keys())[1:]:
            new_dict.update(dict[key])
        new_data.append(new_dict)

    #create df
    df = pd.DataFrame(new_data)

    #rescale
    rescaled_cols = ['bleu', '1-gram-precision', '2-gram-precision', '3-gram-precision', '4-gram-precision', 'meteor','precision','recall','f1']
    df[rescaled_cols]=df[rescaled_cols].apply(rescale)

    #round values
    df = df.round(4)

    return df

def get_some_scores(df):
    selected_cols = ['model','rougeL', 'bleu', 'meteor', 'f1']
    short_df = df[selected_cols]
    new_col_names = ['Model','ROUGE-L', 'BLEU', 'METEOR', 'BERTSCORE']
    short_df.columns= new_col_names

    return short_df

    

def main():

    #define score paths
    non_per_scores_path = './automatic_scores/non_perspective_scores.json'
    per_scores_path = './automatic_scores/perspective_scores.json'

    #create tables dir and define paths
    table_dir = './tables'
    if not os.path.exists(table_dir): os.mkdir(table_dir)
    all_non_per_scores_path = os.path.join(table_dir,'non_perspective_scores.csv')
    all_per_scores_path = os.path.join(table_dir,'perspective_scores.csv')
    some_non_per_scores_path = os.path.join(table_dir,'some_non_perspective_scores.csv')
    some_per_scores_path = os.path.join(table_dir,'some_perspective_scores.csv')
    

    #read scores
    with open(non_per_scores_path,'r') as non_per_file, open(per_scores_path,'r') as per_file:
        non_per_scores = json.load(non_per_file)['scores']
        per_scores = json.load(per_file)['scores']

    #organize all scores
    all_non_per_scores_df = rearrange(non_per_scores)
    all_per_scores_df = rearrange(per_scores)

    #organize some scores
    some_non_per_scores_df = get_some_scores(all_non_per_scores_df)
    some_per_scores_df = get_some_scores(all_per_scores_df)

    #store all scores
    all_non_per_scores_df.to_csv(all_non_per_scores_path)
    all_per_scores_df.to_csv(all_per_scores_path)
    some_non_per_scores_df.to_csv(some_non_per_scores_path)
    some_per_scores_df.to_csv(some_per_scores_path)


if __name__ == "__main__":
    print('Extracting automatic score tables...')
    main()

