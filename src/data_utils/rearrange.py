import pandas as pd
import json
import numpy as np


def clean_dict(dialogue):
     '''
     From each dialogue take only the message (i.e, turns), the sender info and the triples if any.

     Parameters:
     -dialogue (nested dictionary) : a dictionary represening one dialogue. It contains other dictionaries, where each dictionary contains either a dialogue turn or its metadata.
    
     Returns: 
     -clean_list (list) : a list of dictionaries representing one dialgoue. Each dictionary contains either a dialogue turn and the sender, or the turn's triples, if any.
     '''

     clean_list = []
     for dict in dialogue:
          
          #if the dictionary contains a turn
          if dict.get('message'): 
               clean_dict= {'message':dict['message'],'sender':dict['sender']}
          
          #if the dictionary contains the turn metadata
          else: 
               try:
                    clean_dict = {'triples':dict.get('metadata')['path'][1], 'sender':dict['sender']}
               except KeyError: #if metadata has no triples skip it
                    continue

          #so for each dialogue we create a clean_list
          clean_list.append(clean_dict)


     return clean_list


def rearrange(clean_dialogue, index):
    '''
    Process further each dialogue by giving it an index. Add a "starting entity", which signals the topic, to the first turn of the dialogue, if empty.
    
    
    Parameters:
    -clean_dialogue (a list of dicts):  A list of dictionaries representing one dialogue. Each dictionary contains either a dialogue turn and the sender, or the turn's triples if any.
    
    Returns:
    -rearranged_list (a list of dicts): A list of dictionaries representing one dialogue. Each diactionary contains the dialogue id, one turn, a starting entity, if one, the turn's sender and its triples, if any.

    '''
    rearranged_list = []
    for i,dict in enumerate(clean_dialogue):
        #first turn
        #if dictionary is the first in the dialogue and contains a turn
        #if dictionary is the first in the dialogue but it contains metadata the condition in line 81 
        if dict==clean_dialogue[0] and dict.get('message'): 
            #we get the second dictionary
            next_dict = clean_dialogue[i+1]
            #if the next dictionary is not another turn but containts metadata
            if next_dict.get('triples'): 
                start_ent = ''
                #we access the triples in this metadata dict
                for triple in next_dict['triples']:
                    #we iterate the entities in each triple
                    for ent in triple:
                        #if the entity exists in the turn (it might be that the tripple is misassigned)
                        if ent.lower() in dict['message'].lower(): 
                            #then we define that entity as the starting entity
                            start_ent=ent
                            break
                        else:continue
                    #we break again here so that, if we find the starting entity in the first triple, then we don't have to start iterating the entities of another
                    break
                new_dict = {'dial_id':index ,'message':dict['message'],'start_ent':start_ent,'triples':None,'sender':dict['sender']}
            else: #if the next dictionary is another turn
                new_dict = {'dial_id':index, 'message':dict['message'],'start_ent':"",'triples':None,'sender':dict['sender']}
            
            rearranged_list.append(new_dict)
            continue

        #metadata dicts
        if dict.get('triples'):
            continue

        #rest (message dicts)
        prev_dict = clean_dialogue[i-1]
        #if there is metadata for a turn
        if prev_dict['sender']==dict['sender'] and prev_dict.get('triples'):
            new_dict = {'dial_id':index, 'message':dict['message'],'start_ent':"",'triples':prev_dict['triples'],'sender':dict['sender']}
            rearranged_list.append(new_dict)

        #if no metada for a turn
        else:
            new_dict = {'dial_id':index, 'message':dict['message'],'start_ent':"",'triples':None,'sender':dict['sender']}
            rearranged_list.append(new_dict)
    

    return rearranged_list 


def main(old_data_path, new_data_path):

    data = pd.read_csv(old_data_path)

    #create a list with all the dialogues
    all_data = []
    for row in data['Messages']:
        all_data.append(json.loads(row))

    
    #remove dialogues with only one turn
    for dial in all_data:
        if len(dial)==1:
            all_data.remove(dial)   
    
    
    #apply the functions and add a turn_id to the turn of each dialogue
    new_data = []
    for i,dial in enumerate(all_data):
        try:
            clean_dial = clean_dict(dial)
            rearranged_dial = rearrange(clean_dial,i)
            added_turns = []
            #given that every dict in the rearranged dial corresponds to a turn
            for id,dict in enumerate(rearranged_dial):
                dict['turn_id']=id
                added_turns.append(dict)
        #we take out again the single-turn dialogues that were not removed previously
        except IndexError:
            #we skip single-turn dialogues
            continue
        #new_data is a list of dicts representing the whole dataset §§, where each dict corresponds to a dialogue turn.
        new_data.extend(added_turns)
    

    #store preprocessed data
    data_df = pd.DataFrame.from_dict(new_data)
    turn_col = data_df.pop('turn_id')
    data_df.insert(1,'turn_id',turn_col)
    data_df.to_csv(new_data_path) 
    
