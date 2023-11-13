import pandas as pd
import ast
import os
from sklearn.model_selection import train_test_split
from .perspective_utils import extract_perspective
import random
random.seed(128)


def add_history_cols(df):
    '''
    Adds two new columns to the data representing the past turns and past triples of every data point.
    
    Parameters:
    -data_df (DataFrame) : the rearranged data resulted from preprocess.py
    
    Returns:
    -new_data_df (DataFrame) : the same data enhanced with two more columns:
                                            turn_hist: each row is a list of tuples representing the past turns, where each tuple has the id and the message of the past turn
                                            triple_hist: each row is a list of tuples represenitng the past triples associated with the past turns. Each tuple has the id and a list with the triples of the past turn.

    '''

    knowledge_hist_col = []
    turn_hist_col = []
    #iterate the dial ids
    for dial in df['dial_id'].unique():
        #we create a dialogue df with all the turns having this id
        dial_df = df[df['dial_id']==dial]
        for i,row in dial_df.iterrows():
        
            #for the first turn of each dialoge (i.e., if the turn id = 0)
            if row[1]==0:
                #there is no history
                turn_hist_col.append([])
                knowledge_hist_col.append([])

            #for the rest of the turns
            else:
                #FIRST THE TRIPLES
                knowledge_ls = []

                #we take the star_ent into account for the knowledge history - except from turn 0 all the other turns will have turn 0 into their history
                #if the first turn has a starting entity we include it in the history
                if type(dial_df.iloc[0,3])!=float: knowledge_ls.append((0, [[dial_df.iloc[0,3]] , dial_df.iloc[0,4]]))
                else: knowledge_ls.append((0, [[] , dial_df.iloc[0,4]]))
                
                #for every row that is not the first, we take its triples till but excuding the current row as history. The history of the first row is already added above
                for id, knowledge in zip(dial_df.iloc[1:row[1],1].to_list(), dial_df.iloc[1:row[1],4].to_list()): 
                    knowledge_ls.append((id,knowledge))
                
                knowledge_hist_col.append(knowledge_ls)

                #THEN THE TURNS
                turn_ls = []
                for id,turn in zip(dial_df.iloc[:row[1],1].to_list(), dial_df.iloc[:row[1],2].to_list()):
                    turn_ls.append((id, turn))

                turn_hist_col.append(turn_ls)
    
    df['unstructured_hist'],df['structured_hist'] = turn_hist_col,knowledge_hist_col

    return df


################################################################################################################# 
def normalize_triples(triples_col):
    triples_ls = []
    for item in triples_col:
        #if a value is empty
        if type(item)==float: triples_ls.append([])
        #we turn each row in the triple column into a list 
        else: triples_ls.append(ast.literal_eval(item))
    
    #We reduce the amount of unique triples
    norm_triples_ls = []
    for ls in triples_ls:
        #if the triple list is empty
        if ls == []: 
            norm_triples_ls.append(ls)
        else:
            new_ls = []
            #we iterate the triples in the triples list representing one turn
            for triple in ls:
                if not triple[1].startswith('~'): 
                    new_triple = triple
                else:
                    new_triple = [triple[2],triple[1].strip('~'),triple[0]]
                new_ls.append(new_triple)
            norm_triples_ls.append(new_ls)

    return norm_triples_ls

##################################################################################################################

def remove_empty_str(df):
    '''
    We remove the turns with no structured history in their most recent preceding turn.'''

    empty_ls = []
    for i, row in df.iterrows():
        str_hist = row['fact_structured_hist']
        if not any(char.isalpha() for char in str(str_hist[-1][1])):
            empty_ls.append(row)
    
    empty_df = pd.DataFrame(empty_ls)
    new_df = df.drop(empty_df.index, axis=0)
    
    print('################################')
    print()
    print(f'The dataset is condensed by {(len(empty_df)/len(df)) * 100}%')
    print()
    print('################################')
    print()

    return new_df

##################################################################################################################

def split(all_df):
    '''
    Split dataset into train and test. Shuffle the dialogue ids and assignt 80% to train'''

    unique_dials = list(set(all_df['dial_id']))
    random.shuffle(unique_dials)
    train_dials, dev_dials = train_test_split(unique_dials, test_size=0.2, random_state=128)

    train_df = pd.DataFrame()
    for id in train_dials:
        df = all_df[all_df['dial_id']==id]
        train_df = train_df.append(df)
    train_df = train_df.reset_index(drop=True)

    dev_df = pd.DataFrame()
    for id in dev_dials:
        df = all_df[all_df['dial_id']==id]
        dev_df = dev_df.append(df)
    dev_df = dev_df.reset_index(drop=True)

    return train_df, dev_df


##################################################################################################################


def main(old_data_path, new_data_path, train_p, dev_p, act_classifier):

    #we read the data
    data_df = pd.read_csv(old_data_path, index_col=0)
    
    #normalize_triples
    norm_triples_col = normalize_triples(data_df['triples'])
    data_df.pop('triples')
    data_df.insert(4,'triples',norm_triples_col)

    #add_perspective_col
    print('Extracting perspective...')
    print()
    persp_df = extract_perspective.main(data_df, act_classifier)
    #comb perspective triples with factual triples
    persp_df.apply(lambda row: row['triples'].extend(row['perspective_triples']),axis=1)

    #add history columns
    hist_data_df = add_history_cols(persp_df)

    #remove the first turn of each dialogue. It will still be included in the history of other turns
    zero_ids = hist_data_df[hist_data_df['turn_id']==0].index.to_list()
    removed_zero_df = hist_data_df.drop(zero_ids,axis=0)

    #remove the start_ent column. 
    #This is because all of the valeus are going to be empty now that we removed the first turn of each dialogue
    removed_zero_df.pop('start_ent')

    #keep only the perspective of the most recent turn in the history
    new_str_hist_col = []
    for ex in removed_zero_df['structured_hist']:
        new_ex = extract_perspective.keep_most_recent_prsp(ex)
        new_str_hist_col.append(new_ex)
    removed_zero_df.pop('structured_hist')
    removed_zero_df.insert(7,'structured_hist', new_str_hist_col)

    #create only factual structured history column
    factual_str_hist_col = []
    for ex in removed_zero_df['structured_hist']:
        new_ex = extract_perspective.extract_factual(ex)
        factual_str_hist_col.append(new_ex)
    removed_zero_df.insert(8,'fact_structured_hist',factual_str_hist_col)

    #remove turns without structured history
    final_df = remove_empty_str(removed_zero_df)

    #store the entire dataset
    final_df.to_csv(new_data_path)

    #split and store
    train, dev = split(final_df)
    train.to_csv(train_p)
    dev.to_csv(dev_p)



if __name__ == "__main__":
    print('Running convert_input.py')
    


