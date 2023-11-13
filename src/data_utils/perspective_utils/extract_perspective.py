import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import pipeline
#TO USE THIS MODEL FOLLOW THE INSTRUCTIONS HERE: https://github.com/leolani/cltl-dialogueclassification/tree/main 
from .cltl.dialogue_act_classification.midas_classifier import MidasDialogTagger

#this function was extracted from https://github.com/leolani on 11/10/2023 
def sort_predictions(predictions):
    '''
    Sorts the model predictions from the highest scoring one to the lowest
    '''
    return sorted(predictions, key=lambda x: x['score'], reverse=True)

#############################################################################################################

def extract_emotion(df):
    #we use this model from huggingface https://huggingface.co/bhadresh-savani/bert-base-go-emotion
    model_name = "bhadresh-savani/bert-base-go-emotion"
    emotion_pipeline = pipeline('sentiment-analysis',
                                model=model_name, return_all_scores=True, truncation=True)
    
    turns = df['message']
    emotions_col = []
    #we iterate each turn (i.e., each row in the dataframe)
    for turn in turns:
        #we apply sentence tokenization with nltk
        sents = sent_tokenize(turn)
        turn_emotions = []
        #we apply emotion classification to every sentence
        for sent in sents:
            emotion_labels = emotion_pipeline(sent)
            sorted_emotion_labels = sort_predictions(emotion_labels[0])
            emotion = sorted_emotion_labels[0]['label']
            #we store the most likely emotion of each sentence into a list tha represents the entire turn
            turn_emotions.append(emotion)
        emotions_col.append(set(turn_emotions))
    
    #for every turn/row, we turn its emotions into emotion triples using the sender info for that turn
    emotion_triples_col = []
    for emotions_ls, sender in zip(emotions_col, df['sender']):
        #for each turn we create a list of emotion triples
        emotion_triples_ls = []
        for emotion in emotions_ls:
            triple = [sender, 'emotion', str(emotion)]
            emotion_triples_ls.append(triple)
        emotion_triples_col.append(emotion_triples_ls)
        #now each row/turn has a list of emotion triples
    
    return emotion_triples_col

#############################################################################################################

def extract_dial_act(df, classifier):
    analyzer = MidasDialogTagger(model_path = classifier)
    turns = df['message']
    dial_acts_col = []
    #we iterate each turn (i.e., each row in the dataframe)
    for turn in turns:
        turn_dial_acts = []
        #we apply sentence tokenization with nltk
        sents = sent_tokenize(turn)
        #we apply the dialogue act classifier to every sentence
        for sent in sents:
            dial_act = analyzer.extract_dialogue_act(sent)
            dial_act = dial_act[0].value
            turn_dial_acts.append(dial_act)
        dial_acts_col.append(set(turn_dial_acts))
    
    #make dial_act triples
    act_triples_col = []
    for acts_ls, sender in zip(dial_acts_col, df['sender']):
        #for each turn we create a list of dialogue act triples
        act_triples_ls = []
        for act in acts_ls:
            triple = [sender, 'dialogue-act', str(act)]
            act_triples_ls.append(triple)
        act_triples_col.append(act_triples_ls)
    
    return act_triples_col

##############################################################################################################


def keep_most_recent_prsp(tpl_ls):
    '''
    Keep the perpsective information of only the most recent turn in the history

    ARGS
    tpl_ls (list) : a list of tuples, where each tuple contains triples (factual and perspectival) of one turn in the dialogue history 

    RETURN:
    new_tpl_ls (list) : same as tpl_ls but now only the last tuple contains perspectival triples


    '''
    new_tpl_ls = []
    #if the turn has no triples we keep it empty
    if tpl_ls == [] : new_tpl_ls = tpl_ls
    else:
        #each tuple represents one turn in the dialogue history
        for tuple in tpl_ls:
        #we don't want the most recent tuple
            if tuple[0] == tpl_ls[-1][0]: continue
            else:
                if tuple[0]==0: #some cleaning
                    triple_ls = [tuple[1][0]]
                    triple_ls.extend(tuple[1][1])
                    
                else: triple_ls = tuple[1] #the list of triples for that turn

                new_triple_ls = []       
                for triple in triple_ls:
                    if triple == []: new_triple_ls.append(triple)
                    elif triple[0] != 'user' and triple[0] != 'assistant': new_triple_ls.append(triple)
                    else: continue
  
                new_tpl_ls.append((tuple[0],new_triple_ls))

        #we append the most recent tuple at the end
        #we also clean the most recent tuple if it needs cleaning
        last_tuple = tpl_ls[-1]
        if last_tuple[0] == 0:
            triple_ls = [last_tuple[1][0]]
            triple_ls.extend(last_tuple[1][1])
            last_tuple = (last_tuple[0], triple_ls)
        new_tpl_ls.append(last_tuple)

    return new_tpl_ls

def extract_factual(tpl_ls):
    factual_tpl_ls = []
    if tpl_ls == [] : factual_tpl_ls = tpl_ls
    else:
        for tpl in tpl_ls:
            factual_triples = []
            triple_ls = tpl[1]
            for triple in triple_ls:
                if triple == []: factual_triples.append(triple)
                elif triple[0] != 'user' and triple[0] != 'assistant':
                    factual_triples.append(triple)
                else: continue

            factual_tpl = (tpl[0],factual_triples)
            factual_tpl_ls.append(factual_tpl)
    
    return factual_tpl_ls
   

def main(old_df, classifier):

    #extract emotions
    print()
    print('Extracting emotions...')
    print()
    emotion_col = extract_emotion(old_df)

    #extract dialogue acts
    print()
    print('Extracting dialogue acts...')
    print()
    dial_act_col = extract_dial_act(old_df, classifier)

    #combine perspective_cols
    persp_col = []
    for emotion_ls, act_ls in zip(emotion_col, dial_act_col):
        emotion_ls.extend(act_ls)
        persp_ls = emotion_ls
        persp_col.append(persp_ls)
    
    old_df['perspective_triples'] = persp_col
    new_df = old_df

    return new_df

if __name__ == "__main__":
    print('Running extract_perspective.py')