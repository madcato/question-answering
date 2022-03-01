import pandas as pd
import gzip

# Sample: "<S>What is the capital of France?<SEP>France</S>"
QA_INIT_TOKEN = '<S> '  # start of sentence
QA_END_TOKEN = ' </S>'  # end of the sentence
SEPARATOR_TOKEN = ' <SEP> '  # separator between questions and answers

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def split_data(df):
    questions =  df['question'].tolist()
    answers = df['answer'].tolist()
    return questions, answers

def qa_joiner(pair):
    return QA_INIT_TOKEN  + pair[0] + SEPARATOR_TOKEN + pair[1] + QA_END_TOKEN

def save_list_to_file(list, filename):
    with open(filename, 'w') as f:
        for item in list:
            f.write(item + '\n')
