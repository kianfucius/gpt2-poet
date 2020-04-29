import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import logging
import warnings
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import os
import csv
from gpt2_poet import decoders, analysis

try:
    from google.colab import drive
except:
    pass


logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')

def set_device():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    return device

def download_model(name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(name if name == "gpt2-medium" else "gpt2")
    model = GPT2LMHeadModel.from_pretrained(name if name == "gpt2-medium" else "gpt2")
    model = model.to(set_device())
    return model, tokenizer

class Poet(nn.Module):
    def __init__(self, name='gpt2'):
        super(Poet).__init__()
        self.name = name
        self.model, self.tokenizer = download_model(name=self.name)
        # self.discriminator = get_discriminator()

class PoemDataset(Dataset):
    def __init__(self, poems_path):
        super().__init__()
        self.poem_list = []
        self.end_token = "<|endoftext|>"

        with open(poems_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                poem_str = f"POEM:\n{row[0]}\n{self.end_token}"
                self.poem_list.append(poem_str)

    def __len__(self):
        return len(self.poem_list)
    def __getitem__(self, item):
        return self.poem_list[item]

def finetune(model, tokenizer, dataset, models_folder="trained_models", batch_size=1, epochs=5, learning_rate=0.0001, warmup_steps=5000, max_seq_len=400):
    poem_loader = DataLoader(PoemDataset(dataset), batch_size=1, shuffle=True)

    device = set_device()
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)
    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0

    tmp_poems_tens = None
    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    for epoch in range(epochs):
        print(f"EPOCH {epoch} started" + '=' * 30)

        for idx,poem in enumerate(poem_loader):

            #"Fit as many poem sequences into max_seq_len sequence as possible" logic start ####
            poem_tens = torch.tensor(tokenizer.encode(poem[0])).unsqueeze(0).to(device)
            #Skip sample from dataset if it is longer than max_seq_len
            if poem_tens.size()[1] > max_seq_len:
                continue

            #First poem in seq
            if not torch.is_tensor(tmp_poems_tens):
                tmp_poems_tens = poem_tens
                continue
            else:
                #The next poem does not fit in so we process the sequence and leave the last poem
                #as the start for next sequence
                if tmp_poems_tens.size()[1] + poem_tens.size()[1] > max_seq_len:
                    work_poems_tens = tmp_poems_tens
                    tmp_poems_tens = poem_tens
                else:
                    #Add the poem to sequence, continue and try to add more
                    tmp_poems_tens = torch.cat([tmp_poems_tens, poem_tens[:,1:]], dim=1)
                    continue
            #Sequence ready, pass through model

            outputs = model(work_poems_tens, labels=work_poems_tens)
            loss, logits = outputs[:2]
            loss.backward() #auto differentiation ~ lookup
            sum_loss = sum_loss + loss.detach().data

            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == batch_size:
                proc_seq_count = 0
                batch_count += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(models_folder, f"gpt2_poet_epoch_{epoch}.pt"))

def generate(model, tokenizer, model_epoch, num_poems=1000, models_folder="trained_models", output_folder="outputs", file_type="txt", print_output=False, greedy=False):
    device = set_device()

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    model_path = os.path.join(models_folder, f"gpt2_poet_epoch_{model_epoch}.pt")
    model.load_state_dict(torch.load(model_path))

    poems_output_file_path = os.path.join(output_folder, f'generated_{model_epoch}.{file_type}')

    model.eval()
    if os.path.exists(poems_output_file_path):
        os.remove(poems_output_file_path)

    poem_num = 0
    with torch.no_grad():
            for poem_idx in range(num_poems):
                cur_ids = torch.tensor(tokenizer.encode("POEM:")).unsqueeze(0).to(device)

                for i in range(100):
                    outputs = model(cur_ids, labels=cur_ids)
                    cur_ids, poem_finished = decoders.decode(tokenizer, device, cur_ids, outputs, i, greedy)
                    if poem_finished:
                        break

                if poem_finished:
                    poem_num += 1

                    output_list = list(cur_ids.squeeze().to('cpu').numpy())

                    output_text = tokenizer.decode(output_list, skip_special_tokens=True)
                    if print_output:
                        print(f"{output_text[6:]}\n")

                    with open(poems_output_file_path, 'a') as f:
                        if (file_type == 'csv'):
                            f.write(f"\"{output_text[6:]}\n\"\n")
                        else:
                            f.write(f"{output_text[5:]}\n")

def generate_wrapper(model, tokenizer, model_epoch, num_poems=1000, models_folder="trained_models", output_folder="wrapper_outputs", file_type="txt", print_output=False,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        attention_mask=None,
        decoder_start_token_id=None
    ):

    device = set_device()

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    model_path = os.path.join(models_folder, f"gpt2_poet_epoch_{model_epoch}.pt")
    model.load_state_dict(torch.load(model_path))

    poems_output_file_path = os.path.join(output_folder, f'generated_{model_epoch}.{file_type}')

    model.eval()
    if os.path.exists(poems_output_file_path):
        os.remove(poems_output_file_path)

    with torch.no_grad():
        cur_ids = torch.tensor(tokenizer.encode("POEM:")).unsqueeze(0).to('cuda')

    outputs = model.generate(
        input_ids=cur_ids,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample,
        early_stopping=early_stopping,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        bad_words_ids=bad_words_ids,
        bos_token_id=bos_token_id,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_return_sequences=num_poems,
        attention_mask=attention_mask,
        decoder_start_token_id=decoder_start_token_id
    )

    with open(poems_output_file_path, 'a') as f:
        for i in range(num_poems):
            output_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            if print_output:
                print(f"{output_text[6:]}\n")
            if (file_type == 'csv'):
                f.write(f"\"{output_text[6:]}\n\"\n")
            else:
                f.write(f"{output_text[5:]}\n")
