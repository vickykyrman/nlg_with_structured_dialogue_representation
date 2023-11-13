import re

def process_input(data, model_tokenizer, model_args):
    #we use padding in our case with max target length = 75
    padding = "max_length" if model_args.pad_to_max_length else False
    max_target_length = model_args.max_target_length

    def dataset_mapping_function(examples):
        contexts = examples['Context']
        responses = examples['Response']
        kbs = examples['Knowledge']
    
        inputs = []
        for context, kb in zip(contexts, kbs):
            if model_args.no_kb:
                inputs.append(context + ' => ')
            #in our case always the following applied. 
            else:
                _input = context + ' <|Knowledge|> ' + kb + ' => '
                inputs.append(_input)
        model_inputs = model_tokenizer(inputs, max_length=model_args.max_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets/predictions
        with model_tokenizer.as_target_tokenizer():
            labels = model_tokenizer(responses, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        #In our case this condition applies
        if padding == "max_length" and model_args.ignore_pad_token_for_loss:
            labels["labels"] = [
                [(l if l != model_tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["labels"]
        return model_inputs
    
    
    #prep data
    column_names = ['Context','Response','Knowledge']
    processed_data = data.map(
        dataset_mapping_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc=f"Processing dataset",
    )
 
    return processed_data

###################################################################################################
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



def post_process_text(preds, labels):
    preds = [normalize_answer(pred.strip()) for pred in preds]
    labels = [normalize_answer(label.strip()) for label in labels]

    return preds, labels    
###################################################################################################      




if __name__=='__main__':
    print('Running process input...')
        




     