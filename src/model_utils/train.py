###The following code was adopted from https://github.com/microsoft/GODEL on 29/3/2023

#import packages
from .process_input import process_input
from accelerate import Accelerator
import transformers
import re
import json
import random
import numpy as np
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
import torch
import logging
from tqdm.auto import tqdm
import os
import math
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


def train(data_path, MODEL_NAME, output_dir):
    global global_kwargs

    ###################################### CONFIGURE

    logger = logging.getLogger(__name__)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        )
    logger.setLevel(logging.INFO)

    accelerator = Accelerator()
    # if accelerator.is_local_main_process:
    #     datasets.utils.logging.set_verbosity_warning()
    #     transformers.utils.logging.set_verbosity_info()
    # else:
    #     datasets.utils.logging.set_verbosity_error()
    #     transformers.utils.logging.set_verbosity_error()

    extension = 'json'
    train_data = load_dataset(extension, data_files=data_path)

    config = AutoConfig.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/GODEL-v1_1-base-seq2seq')
    model = AutoModelForSeq2SeqLM.from_pretrained(
            'microsoft/GODEL-v1_1-base-seq2seq',
            from_tf=bool(".ckpt" in 'microsoft/GODEL-v1_1-base-seq2seq'),
            config=config)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    #we label the pad tokens -100 because we don't want them to be taken into account in the calculation of the loss
    label_pad_token_id = -100 if global_kwargs.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )
    
    #DATA
    #preprocess data 
    prep_train = process_input(train_data, tokenizer, global_kwargs)
    prep_train = prep_train['train']
    train_dataloader = DataLoader(
        prep_train, shuffle=True, collate_fn=data_collator, batch_size=global_kwargs.per_device_train_batch_size
    )


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": global_kwargs.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=global_kwargs.learning_rate)

    #uncomment to run locally
    #model.to('cpu')

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader)
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / global_kwargs.gradient_accumulation_steps)
    total_training_steps = global_kwargs.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=global_kwargs.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=global_kwargs.num_warmup_steps,
        num_training_steps=total_training_steps
    )

    ##################################### TRAIN
    total_batch_size = global_kwargs.per_device_train_batch_size * accelerator.num_processes * global_kwargs.gradient_accumulation_steps

    logger.info("***** TRAINING... *****")
    logger.info(f"  Num training examples = {len(prep_train)}")
    logger.info(f"  Num Epochs = {global_kwargs.num_train_epochs}")
    logger.info(f"  Instantaneous batch size = {global_kwargs.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {global_kwargs.gradient_accumulation_steps}")

    #set progress bar and start training
    progress_bar = tqdm(total=len(train_dataloader) * global_kwargs.num_train_epochs, dynamic_ncols=True)
    model.train()
    # Reset the loss accumulator for the next logging interval
    tr_loss = 0.0
    for epoch in range(global_kwargs.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / global_kwargs.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % global_kwargs.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            if progress_bar.n >= progress_bar.total:
                break

            if (step + 1) % global_kwargs.logging_steps == 0:
                average_loss = tr_loss / global_kwargs.logging_steps  # Calculate average loss
                logger.info(f"AVERAGE LOSS: {average_loss}")
    
    #store model
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    accelerator.save(optimizer.state_dict(), os.path.join(output_dir,"optimizer.bin"))
    torch.save(global_kwargs, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving final model checkpoint to %s", output_dir)
    progress_bar.close()


###################################################################################################################

def main(data_dr, models_dr, qual_scenario, quant_scenario, kwargs, perspective = None):
    global global_kwargs
    global_kwargs = kwargs

    if perspective == None:
        train_path = os.path.join(data_dr,f'non_perspective/{qual_scenario}/{qual_scenario}_{quant_scenario}_train.json')
        model_name = f'Godel-{qual_scenario}-{quant_scenario}'
    else: 
        train_path = os.path.join(data_dr,f'perspective/{qual_scenario}/{qual_scenario}_{quant_scenario}_train.json')
        model_name = f'Godel-{qual_scenario}-Per-{quant_scenario}'
    
    model_dr = os.path.join(models_dr, model_name)
    train(train_path, model_name, model_dr)



if __name__=='__main__':
    print('Running train.py...')