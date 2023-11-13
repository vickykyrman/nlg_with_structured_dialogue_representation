###The following code was adopted from https://github.com/microsoft/GODEL on 29/3/2023

#import packages
from .process_input import process_input, post_process_text
import transformers
import re
import json
import random
import numpy as np
import datasets
from datasets import load_dataset, load_metric
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
import torch
import logging
from tqdm.auto import tqdm
import os
import math
from torch.nn.functional import pad
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_scheduler,
    set_seed,
)

def store_scores(dc,path):
    #if there is not file yet
    if not os.path.exists(path):
        with open(path,'w') as file:
            data = {'scores':[]}
            data['scores'].append(dc)
            json.dump(data, file, indent=4)
    #if the file exists        
    else:
        with open(path,'r') as infile:
            data = json.load(infile)
        data['scores'].append(dc)
        
        with open(path, 'w') as outfile:
            json.dump(data,outfile,indent=4)


def test(data_path, MODEL_NAME, model_path, scores_path, preds_path):
    global global_kwargs

    ###################################### CONFIGURE
    accelerator = Accelerator()

    logger = logging.getLogger(__name__)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        )
    logger.setLevel(logging.INFO)

    #load model
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    #uncomment to run locally
    #model.to('cpu')

    gen_kwargs = {
                "max_length": global_kwargs.max_length,
                "num_beams": global_kwargs.num_beams,
         }

    #DATA
    dev_data = load_dataset('json', data_files=data_path)
    prep_dev = process_input(dev_data, tokenizer, global_kwargs)
    prep_dev = prep_dev['train'] #but in realiy it's the dev

    #we label the pad tokens -100 because we don't want them to be taken into account in the calculation of the loss
    label_pad_token_id = -100 if global_kwargs.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )

    dev_dataloader = DataLoader(prep_dev, shuffle=False, collate_fn=data_collator, batch_size=global_kwargs.per_device_eval_batch_size)
    
    model, tokenizer, dev_dataloader = accelerator.prepare(model, tokenizer, dev_dataloader)
    
    #METRICS
    metric_rouge = load_metric("rouge")
    metric_bleu = load_metric("bleu")
    metric_meteor = load_metric("meteor")
    metric_bertscore = load_metric("bertscore")

    ##################################### EVALUATE

    logger.info("***** EVALUATING... *****")
    logger.info(f"  Num validation examples = {len(prep_dev)}")
    progress_bar = tqdm(total=len(dev_dataloader), dynamic_ncols=True)
    model.eval()

    decoded_preds_list = []
    for step, batch in enumerate(dev_dataloader):
        with torch.no_grad():
            # Get the predicted tokens
            generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                    )

            generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )
            labels = batch["labels"]
            
            if not global_kwargs.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
            
            #predictions
            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            #gold
            labels = accelerator.gather(labels).cpu().numpy()

            if global_kwargs.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


            #normalize predictions and labels for evaluation
            norm_preds, norm_labels = post_process_text(decoded_preds, decoded_labels)
            decoded_preds_list.extend(norm_preds)
        
            progress_bar.update(1)

            #apply metrics per batch
            metric_rouge.add_batch(predictions=norm_preds, references=norm_labels)
            metric_meteor.add_batch(predictions=norm_preds, references=norm_labels)
            metric_bertscore.add_batch(predictions=norm_preds, references=norm_labels)
            _norm_preds = [i.split() for i in norm_preds]
            _norm_labels = [[i.split()] for i in norm_labels]
            metric_bleu.add_batch(predictions=_norm_preds, references=_norm_labels)
    
    logger.info("###############   SCORES   ###################")
    result = metric_rouge.compute(use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    logger.info(result)
    print()
    print()
    result_bleu = metric_bleu.compute()
    logger.info(result_bleu)
    print()
    print()

    result_meteor = metric_meteor.compute()
    logger.info(result_meteor)
    print()
    print()

    result_bertscore = metric_bertscore.compute(lang="en", model_type="distilbert-base-uncased")
    result_bertscore['precision'] = np.average(result_bertscore['precision'])
    result_bertscore['recall'] = np.average(result_bertscore['recall'])
    result_bertscore['f1'] = np.average(result_bertscore['f1'])
    logger.info(result_bertscore)
    print()
    print()

    scores_dict = {'model': MODEL_NAME, 'ROUGE': result, 'BLEU': result_bleu, 'METEOR': result_meteor, 
                       'BERTSCORE': result_bertscore}
    
    #store scores
    store_scores(scores_dict,scores_path)

    #store predictions
    json.dump(decoded_preds_list, open(preds_path,'w'), indent=2)
    logger.info("Saving model outputs to %s", preds_path)





def main(data_dr, models_dr, qual_scenario, quant_scenario, kwargs, all_scores_dir, perspective = None):
    global global_kwargs
    global_kwargs = kwargs

    if perspective == None:
        dev_path = os.path.join(data_dr,f'non_perspective/{qual_scenario}/{qual_scenario}_{quant_scenario}_dev.json')
        model_name = f'Godel-{qual_scenario}-{quant_scenario}'
        scores_dir = os.path.join(all_scores_dir,'non_perspective_scores.json')
    else: 
        dev_path = os.path.join(data_dr,f'perspective/{qual_scenario}/{qual_scenario}_{quant_scenario}_dev.json')
        model_name = f'Godel-{qual_scenario}-Per-{quant_scenario}'
        scores_dir = os.path.join(all_scores_dir,'perspective_scores.json')
    
    model_dr = os.path.join(models_dr, model_name)
    predictions_folder = os.path.join(data_dr,'predictions')
    if not os.path.exists(predictions_folder): os.mkdir(predictions_folder)
    predictions_dir = os.path.join(predictions_folder,f'{model_name}_preds')

    test(dev_path,model_name, model_dr, scores_dir,predictions_dir)


        
if __name__=='__main__':
    print('Running test.py...')


