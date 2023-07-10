# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
import json
import torch
from torch.nn.utils.rnn import pad_sequence

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 10

import os
import csv
import re

SEPARATOR = '->'
PAD_IDX = 1

def separate_first_word(text):
    # Find the index of the first space character
    space_index = text.find(' ')
    if space_index == -1:
        return text, ''
    
    # Extract the first word
    first_word = text[:space_index]
    
    # Find the index of the opening parenthesis
    opening_parenthesis_index = text.find('(')
    
    # Check if there is an opening parenthesis and it occurs before the first space
    if opening_parenthesis_index != -1 and opening_parenthesis_index < space_index:
        # Find the index of the closing parenthesis
        closing_parenthesis_index = text.find(')', opening_parenthesis_index)
        
        # Check if there is a closing parenthesis
        if closing_parenthesis_index != -1:
            # Extract the text after the closing parenthesis
            rest = text[closing_parenthesis_index + 1:].strip()
        else:
            # No closing parenthesis found, consider everything after the first space as the rest
            rest = text[space_index + 1:].strip()
    else:
        # No opening parenthesis found before the first space, consider everything after the first space as the rest
        rest = text[space_index + 1:].strip()
    
    # Remove parentheses and their contents from the rest
    rest = re.sub(r'\([^()]*\)', '', rest)
    
    return first_word, rest

def load_csv_files(directory):
    data_list = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", newline='', encoding='latin-1') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if row:
                        first_word, rest = separate_first_word(row[0])
                        otuput_text = first_word.strip() + SEPARATOR + rest.strip()
                        data_list.append([first_word.strip(), otuput_text])
                        data_list.append([rest.strip(), otuput_text])
    
    return data_list

# Specify the subdirectory name
subdirectory = "data"

# Call the function to load CSV files and retrieve the list of arrays
data = load_csv_files(subdirectory)

# Print the resulting list of arrays
# for entry in data:
#     print(entry)

dataset = Dataset.from_dict({'text': data})

# owt by default only contains the 'train' split, so create a test split
split_dataset = dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
split_dataset['validation'] = split_dataset.pop('test') # rename the test split to val

# this results in:
# >>> split_dataset
# DatasetDict({
#     train: Dataset({
#         features: ['text'],
#         num_rows: 8009762
#     })
#     val: Dataset({
#         features: ['text'],
#         num_rows: 4007
#     })
# })

# def save_text_file(file_path, string_list):
#     with open(file_path, 'w') as file:
#         for string in string_list:
#             file.write(string + '\n')


types = ["train", "validation"]
for type in types:
    output_arr = []
    dataset = split_dataset[type]
    for split in dataset:
        src = split['text'][0]
        tgt = split['text'][1]
        output_arr.append({ 'translation': { "input": src, 
                            "output": tgt
                          }})

    filename = os.path.join(os.path.dirname(__file__), f'{type}.json')
    print(f'writing {filename}')
    with open(filename, 'w') as file:
        json.dump(output_arr, file)
