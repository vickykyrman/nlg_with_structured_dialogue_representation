import pandas as pd
import json
import re
import os
import random
import numpy as np
import random
import csv

np.random.seed(128)
random.seed(128)

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    re_art = re.compile(r'\b(a|an|the)\b')
    re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s

###############################################################################

def get_reference(c_path):
    #read context file
    with open(c_path,'r') as infile: 
        dev = json.load(infile)
    #get norm references and contexts
    references = []
    contexts = []
    for dc in dev:
        contexts.append(dc['Context'])
        reference = dc['Response']
        reference = normalize_answer(reference.strip())
        references.append(reference)
    
    return contexts, references

###############################################################################

def extract_data(dir):
    '''
    Extract the predictions from the models we want to annotate and the references that need to be annotated as well. 
    Models are given an aphabetical id for an unbiased annotation process.
    '''

    #get predictions dir
    preds_dir = os.path.join(dir,'predictions')
    context_dir = os.path.join(dir,'non_perspective/Un/Un_All_dev.json')
    #we choose the best performing models and their perspective model variants
    wanted = ['Godel-Str-Shared','Godel-Comb-Half', 'Godel-Un-Half', 'Godel-Comb-Per-Half','Godel-Str-Per-Shared']
    alpha = ['A', 'B', 'C', 'D', 'E', 'F']
    i = 0
    preds_dc = {}
    chars_dc = {}
    for file_name in os.listdir(preds_dir):
        model_name = file_name.replace('_preds',"")
        if model_name in wanted: 
            path = os.path.join(preds_dir,file_name)
            with open(path, 'r') as infile:
                data = json.load(infile)
                preds_dc[model_name]=data
                chars_dc[model_name] = alpha[i]
                i+=1
        else: continue
    
    c,r = get_reference(context_dir)
    preds_dc['reference'] = r
    chars_dc['reference'] = alpha[i]
            
    return preds_dc, chars_dc, c

#################################################################################################

def read_dev():
    dev_dc = {}
    wanted_dev_paths = {'Str-Shared':'../../data/non_perspective/Str/Str_Shared_dev.json',
                        'Comb-Half':'../../data/non_perspective/Comb/Comb_Half_dev.json', 
                        'Un-Half':'../../data/non_perspective/Un/Un_Half_dev.json', 
                        'Comb-Per-Half':'../../data/perspective/Comb/Comb_Half_dev.json',
                        'Str-Per-Shared':'../../data/perspective/Str/Str_Shared_dev.json'}
    
    for name,path in wanted_dev_paths.items():
        with open(path,'r')as infile:
            data = json.load(infile)
        dev_dc[name]=data
        
    return dev_dc

#################################################################################################

def filter_ids(amount, str_dev):
    '''
    filtering
    # 10 X 2 triples
    # 10 X 3 triples
    # 30 X (3 < triples)
    '''

    #FILTER IDS (index ids in the dataset. NOT turn or dialogue ids)
    two_triple_ids = []
    three_triple_ids = []
    more_triple_ids = []
    total_ids = []
    while len(total_ids) != amount:
        #we pick up a random id
        id = random.randint(0,len(str_dev)-1) #it doesn't matter which scenario we put here because we want only the ids which are the same across scenarios
        triples = str_dev[id]['Knowledge'].split('.')
        ###############we want all of the picked instances to have more than one factual triples
        if all([len(triples)==2, len(two_triple_ids)<10, id not in two_triple_ids]): 
            two_triple_ids.append(id)
        elif all([len(triples)==3, len(three_triple_ids)<10, id not in three_triple_ids]):
            three_triple_ids.append(id)
        elif all([len(triples)>3, len(more_triple_ids)<30, id not in more_triple_ids]):
            more_triple_ids.append(id)
        else: 
            total_ids = two_triple_ids + three_triple_ids + more_triple_ids
            continue

    print(f'Number of turns per model after filtering: {len(total_ids)}')
    return total_ids

#################################################################################################

def get_trial_ids(original,str_dev):
    ids = []
    while len(ids) < 10: 
        id = random.randint(0,len(str_dev)+1)
        if id not in original: 
            ids.append(id)
    return ids

#################################################################################################

def prepare_files(ids, contexts, preds_dc, chars_dc, dev_dc):
    '''Create annotation files and clue files'''

    criteria = [{'soundness':0, 
             'conciseness':0, 
             'completeness':0, 
             'relevance':0, 
             'clarity':0, 
             'brevity':0, 
             'coherence':0, 
             'dialogue_act':"",
             'emotion':"",
             'communicative_goal':""}]
    
    #create clues
    clues = []
    for id in ids:
    #we store the dialogue context separately since in some scenarios there is not unstructured dialogue context
        context_ls = contexts[id].split(' EOS ')
        new_context_ls = []
        for i, turn in enumerate(context_ls):
            turn_str = f'<br><br> [{i}] {turn} '
            new_context_ls.append(turn_str)
        context = "".join(new_context_ls)
        id_dicts = []
        for key in preds_dc.keys(): #we iterate the models
            dc = {}
            dc['model_id'] = chars_dc[key]
            dc['dialogue'] = context
            dc['data_id'] = id
            if key=='reference':
                dc['structured_history']=""
                dc['unstructured_history']=""
            else:
                dev_key = key.replace('Godel-',"")
                dc['structured_history'] = dev_dc[dev_key][id]['Knowledge']
                dc['unstructured_history'] = dev_dc[dev_key][id]['Context']
            dc['response'] = preds_dc[key][id]
            dc['reference'] = preds_dc['reference'][id]
            dc['annotations'] = criteria
            dc['comments'] = ""
            id_dicts.append(dc)

        random.seed(128)
        random.shuffle(id_dicts)
        clues.extend(id_dicts)

    #create annotations
    annotations = []
    for dc in clues:
        new_dc = {}
        for key, value in dc.items():
            if all([key!='structured_history', key!='unstructured_history', key!='reference']):
                new_dc[key]=value
        annotations.append(new_dc)
    
    return clues, annotations

#################################################################################################

def store_files(dir,clues,annotations,trial=False):
    if trial==False: clues_name,annotations_name = 'clues','annotations'
    else: clues_name,annotations_name = 'trial_clues','trial_annotations'

    #define paths
    clues_csv = os.path.join(dir,f'{clues_name}.csv')
    annotations_csv = os.path.join(dir,f'{annotations_name}.csv')
    annotations_jsonl = os.path.join(dir,f'{annotations_name}.jsonl')

    #save files
    cols = annotations[0].keys()
    with open(annotations_csv,'w',newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=cols)
        writer.writeheader()
        writer.writerows(annotations)
    
    with open(annotations_jsonl, "w") as file:
        for dc in annotations:
            json_line = json.dumps(dc)
            file.write(json_line + "\n")
    
    cols = clues[0].keys()
    with open(clues_csv,'w', newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=cols)
        writer.writeheader()
        writer.writerows(clues)

#################################################################################################
    
def main():
    data_dir = '../../data'
    manual_data_dir = './data'

    if not os.path.exists(manual_data_dir): os.mkdir(manual_data_dir)

    #extract preds and ref
    preds_dict,chars_dict,context_ls = extract_data(data_dir)
    #store_chars
    chars_path = os.path.join(manual_data_dir,'chars.json')
    json.dump(chars_dict, open(chars_path, 'w'), indent=2)
    
    #extract dev
    dev_dict = read_dev()

    #calculate num of turns to annotate
    num_annotated_models = len(dev_dict.keys())+1
    num_turns_per_model = 50
    total_num_turns = num_turns_per_model * num_annotated_models
    print(f'Total number of turns to annotate: {total_num_turns}')
    print()
    print(f'Number of turns to annotate per model: {num_turns_per_model}')
    print()

    #get ids
    str_info = dev_dict['Str-Shared']
    original_ids = filter_ids(num_turns_per_model,str_info)
    trial_ids = get_trial_ids(original_ids,str_info)

    #preprare annotations and clues files
    original_clues,original_annotations = prepare_files(original_ids,context_ls,preds_dict,chars_dict,dev_dict)
    trial_clues, trial_annotations = prepare_files(trial_ids,context_ls,preds_dict,chars_dict,dev_dict)

    #store files
    store_files(manual_data_dir,original_clues,original_annotations,trial=False)
    store_files(manual_data_dir,trial_clues,trial_annotations,trial=True)

    print(f'Files stored in {os.path.abspath(manual_data_dir)}')

if __name__=="__main__":
    print('Creating annotation files...')
    main()