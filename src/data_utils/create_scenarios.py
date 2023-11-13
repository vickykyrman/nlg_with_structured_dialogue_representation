import pandas as pd
import ast
import os
import json
import spacy
nlp = spacy.load('en_core_web_sm')
import neuralcoref
neuralcoref.add_to_pipe(nlp)

def set_quantity(row, quantity_scenario, quality_scenario, str_col, un_col):
    #-------------------------------------------------------------------------------------------
    #history contains all the past turns
    if quantity_scenario=='all':
        if quality_scenario=='Comb' or quality_scenario=='Str':
            #KNOWLEDGE
            #a list of strings. Each string inludes triple sentences of a each history turn.
            knowledge_ls = []
            for tuple in row[str_col]: #each knowledge tuple corresponds to a past turn
                #a list of string triples
                triple_str_list = []
                for triple in tuple[1]: #a list of triples
                    if triple==[]:continue
                    else:
                        #if triple is a list of triples itself (it happens in very rare cases)
                        if type(triple[0])==list:
                            for tr in triple:
                                triple_str_list.append(" ".join(tr))
                        else: triple_str_list.append(" ".join(triple))

                #we turn that list into a string of sentences, where each sentence is a triple
                triple_str = ". ".join(triple_str_list)
                #it could be that there is not structured history in a history turn
                if triple_str == "": continue
                else: knowledge_ls.append(triple_str)

            
            knowledge_str = '. '.join(knowledge_ls)
            if quality_scenario=='Str': context_str = ""

        if quality_scenario=='Comb' or quality_scenario=='Un':
            #CONTEXT
            history_str_ls = [turn for id, turn in row[un_col]]
            context_str = ' EOS '.join(history_str_ls)
            if quality_scenario=='Un': knowledge_str = ""
    #------------------------------------------------------------------------------------------------------
    #history contains only the most recent turn
    elif quantity_scenario=='one':
        if quality_scenario=='Comb' or quality_scenario=='Str':
            #KNOWLEDGE
            if row[str_col]==[]: knowledge_str = ""
            else:
                last_tuple = row[str_col][-1]
                triple_str_list = []
                for triple in last_tuple[1]: #a list of triples
                    if triple==[]:continue
                    else:
                        #if triple is a list of triples itself (it happens in very rare cases)
                        if type(triple[0])==list:
                            for tr in triple:
                                triple_str_list.append(" ".join(tr))
                        else: triple_str_list.append(" ".join(triple))
                knowledge_str = ". ".join(triple_str_list)
            if quality_scenario=='Str': context_str = ""
        
        if quality_scenario=='Comb' or quality_scenario=='Un':
            #CONTEXT
            if row[un_col]==[]:
                context_str = ""
            else: 
                last_tuple = row[un_col][-1]
                context_str = last_tuple[1]
            if quality_scenario=='Un': knowledge_str = ""

    #--------------------------------------------------------------------------------------------------
    #history contains half of the past turns
    elif quantity_scenario=='half':
        if quality_scenario=='Comb' or quality_scenario=='Str':
            #KNOWLEDGE
            half = round(len(row[str_col])/2)
            knowledge_ls = []
            for tuple in row[str_col][-half:]: #each knowledge tuple corresponds to a past turn
                triple_str_list = []
                for triple in tuple[1]: #a list of triples
                    if triple==[]:continue
                    else:
                        #if triple is a list of triples itself (it happens in very rare cases)
                        if type(triple[0])==list:
                            for tr in triple:
                                triple_str_list.append(" ".join(tr))
                        else: triple_str_list.append(" ".join(triple))

                #we turn that list into a string of sentences, where each sentence is a tuple
                triple_str = ". ".join(triple_str_list)
                if triple_str == "": continue
                else: knowledge_ls.append(triple_str)
        
            knowledge_str = '. '.join(knowledge_ls)
            if quality_scenario=='Str': context_str = ""
        
        if quality_scenario=='Comb' or quality_scenario=='Un':
        #CONTEXT
            half = round(len(row[un_col])/2)
            history_str_ls = [turn for id, turn in row[un_col][-half:]]
            context_str = ' EOS '.join(history_str_ls)
            if quality_scenario=='Un': knowledge_str = ""

    #------------------------------------------------------------------------------------------------
    elif quantity_scenario=='shared':
        
        if quality_scenario=='Un':
            #CONTEXT
            wanted_turns = []

            if row[un_col]!=[]:
                context_list = [tpl[1] for tpl in row[un_col]]
                to_resolve_str = ' EOS '.join(context_list)
                resolved_context = nlp(to_resolve_str)._.coref_resolved
                #a list that contains all the history turns of the current turn with correference resolved.
                resolved_context_list = resolved_context.split(' EOS ')
                last_resolved_turn = resolved_context_list[-1]
    
                target_ents = [token.text for token in nlp(last_resolved_turn) if token.pos_=='PROPN']

                #we include any past turns that share at least one common entity with the most recent turn
                for orig_turn, cor_turn in zip(context_list[:-1], resolved_context_list[:-1]):
                    turn_ents = [token.text for token in nlp(cor_turn) if token.pos_=='PROPN']
                    if any(ent in target_ents for ent in turn_ents):
                        wanted_turns.append(orig_turn)
                    else: continue

                #we include to history necessarily the most recent turn 
                wanted_turns.append(context_list[-1])

                context_str = ' EOS '.join(wanted_turns)
 
            else: context_str = ""

            knowledge_str=""

        elif quality_scenario=='Str':
            #KNOWLEDGE
            
            if row[str_col]!=[]:
                knowledge_ls = []

                target_entities = []  #a list with all the entities of the most_recent turn, as extracted from its triples
                recent_triples_ls = []
                #for the most recent turn
                for triple in row[str_col][-1][1]: 
                    if triple==[]:continue
                    elif type(triple[0])==list: 
                        for tr in triple: recent_triples_ls.append(tr)
                    else: recent_triples_ls.append(triple)
                for triple in recent_triples_ls:
                    target_entities.extend([ent for ent in triple])
                recent_triple_str_list = [" ".join(tr) for tr in recent_triples_ls]
                recent_triple_str = ". ".join(recent_triple_str_list)
                
            

                for tpl in row[str_col][:-1]: #we iterate knowledge history
                    triples_ls = []
                    hist_entities = []
                    for triple in tpl[1]:
                        if triple==[]:continue
                        elif type(triple[0])==list: 
                            for tr in triple: triples_ls.append(tr)
                        else: triples_ls.append(triple)
        
                    for triple in triples_ls:
                        hist_entities.extend([ent for ent in triple])
                        if any(ent in target_entities for ent in hist_entities):
                            triple_str_list = [" ".join(tr) for tr in triples_ls]
                            triple_str = ". ".join(triple_str_list)
                            if triple_str =="":continue
                            else: knowledge_ls.append(triple_str)
                        else: continue

                if recent_triple_str !="":knowledge_ls.append(recent_triple_str)

                knowledge_str = '. '.join(knowledge_ls)
            
            else: knowledge_str = ""
            
            context_str = ""
            

    return context_str, knowledge_str


def set_quality(df, str_c, un_c):
    '''
    Turn the input according to the quantitative and qualitative scenarios and in a format processable by the model.
    
    Parameters:
    -df (DataFrame) : The train or test data
    -str_c (string) : The name of the structured history column
    -un_c (string)  : The name of the unstructured history column
    
    Returns:
    (dictionary) :  a dictionary corresponding to the whole dataset. It includes 4 lists correspoding to the 4 quantitative scenaria. 
                    Each of the 4 lists includes dictionaries as many as the turns in the data. Each dictionary includes:
                            -Context: A turn string with all the history turns separated by ' EOS '
                            -Knowledge: A triples string separated by '' to distinguish among history turns
                            -Response: The reference response string
    '''
    
    qualities = ['Str', 'Un', 'Comb']
    qualities_dc = {}
    for quality in qualities:
        print()
        print(f'Creating {quality} scenario...')
        all = []
        half = []
        one = []
        shared = []
        for i,row in df.iterrows():
            print('Creating ONE...')
            print()
            hist_context_one, hist_knowledge_one = set_quantity(row, 'one', quality, str_c, un_c)
            print('Creating ALL...')
            print()
            hist_context_all, hist_knowledge_all = set_quantity(row, 'all', quality, str_c, un_c)
            print('Creating HALF...')
            print()
            hist_context_half, hist_knowledge_half = set_quantity(row, 'half', quality, str_c, un_c)

            one.append({'Context':hist_context_one, 'Knowledge':hist_knowledge_one, 'Response':row[2]})
            all.append({'Context':hist_context_all, 'Knowledge':hist_knowledge_all, 'Response':row[2]})
            half.append({'Context':hist_context_half,'Knowledge':hist_knowledge_half, 'Response':row[2]})

            if quality == 'Comb': continue
            else: 
                print('Creating SHARED...')
                print()
                hist_context_shared, hist_knowledge_shared = set_quantity(row, 'shared', quality, str_c, un_c)
                shared.append({'Context':hist_context_shared, 'Knowledge':hist_knowledge_shared, 'Response':row[2]})
            
        if quality == 'Comb':
            input_dc = {f'{quality}_All': all, f'{quality}_Half': half, f'{quality}_One' : one}
    
        else: 
            input_dc = {f'{quality}_All': all, f'{quality}_Half': half, f'{quality}_One' : one, f'{quality}_Shared' : shared }
        
        qualities_dc[quality] = input_dc
    
    return qualities_dc

def main(path, perspective=False):

    df = pd.read_csv(path, index_col=0)

    if perspective == True:
        structured_col = 'structured_hist'
    else: 
        structured_col = 'fact_structured_hist'
    
    new_triples_ls = []
    for triple in df[structured_col]:
        #if a value is empty
        if type(triple)==float: new_triples_ls.append([])
        #we turn each row in the triple column into a list 
        else: new_triples_ls.append(ast.literal_eval(triple))
    
    df.pop(structured_col)
    df.insert(len(df.columns), structured_col,new_triples_ls)

    unstructured_col = 'unstructured_hist'
    new_context_ls = []
    for context in df[unstructured_col]:
        new_context_ls.append(ast.literal_eval(context))
    
    df.pop(unstructured_col)
    df.insert(len(df.columns)-2, unstructured_col,new_context_ls)

    qual_dc = set_quality(df, structured_col, unstructured_col)

    return qual_dc



  

if __name__ == "__main__":
    print('Running create_scenarios.py')
