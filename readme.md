# Question anwsering

## Install 
0. Install torch==1.9.0 and torchtext==0.10.0 for 
1. `git clone git@gitlab:ai/question-answering.git`
2. `cd question-answering`
3. `git submodule init`
4. `git submodule update`
5. `source .venv/bin/activate`
6. `cd transformers`
7. `pip3 install -e .`
8. `pip3 install pandas`
9. `pip3 install -U scikit-learn`

## ToDo
- [x] Mirar para aprender https://huggingface.co/transformers/v3.2.0/quicktour.html
- [x] Estudiar https://huggingface.co/transformers/v3.2.0/preprocessing.html)
- [x] Estudiar https://huggingface.co/transformers/v3.2.0/training.html

## Doc
Actually, for solving a **question-answering** problem like the email answering, we must use **text-generation** solutions, the type of task we must use **text2text-generation**. Like:

- [Text Generation](https://huggingface.co/tasks/text-generation)
- [Task Summary: Text Generation](https://huggingface.co/docs/transformers/task_summary#text-generation)
- [Hugging Face: install from source](https://huggingface.co/docs/transformers/installation#installing-from-source)
- [transformers/examples/pytorch/text-generation/](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation)
- [Hugging Face - Text2Text Generation models](https://huggingface.co/models?language=es&library=pytorch&pipeline_tag=text2text-generation&sort=downloads)
- [Huggingface: fine-tuning with custom datasets](https://huggingface.co/transformers/v3.2.0/custom_datasets.html)
- [transformers/examples/pytorch/language-modeling/](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)
- [train transformer with pytorch](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation)
- [Finetune in native pytorch](https://huggingface.co/docs/transformers/master/en/training#finetune-in-native-pytorc)]
- [Preprocessing data](https://huggingface.co/docs/transformers/preprocessing)
- [GPT2Config param doc](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config)
- [transformers.ConversationalPipeline](https://huggingface.co/docs/transformers/v4.17.0/en/main_classes/pipelines#transformers.ConversationalPipeline)
- [DialoGPT](https://github.com/microsoft/DialoGPT)
- [StackOverflow dataset](https://github.com/elastic/rally-tracks/tree/master/so)
- [Text similarity search with vector fields](https://www.elastic.co/blog/text-similarity-search-with-vectors-in-elasticsearch)
- [Huggingface: Preprocessing data](https://huggingface.co/transformers/v3.2.0/preprocessing.html)
- [Huggingdace: Training and fine-tuning](https://huggingface.co/transformers/v3.2.0/training.html) (Aquí explican como crear un fine-tuning custom con PyTorch y Huggingface)
- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER)


## Guide 1 to retrain a GPT-2 model with PyTorch
- [Fine-tuning GPT2 for Text Generation Using Pytorch](https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7)
- [This previous guide uses the old huggingface transformer script](https://github.com/huggingface/transformers/blob/master/examples/legacy/run_language_modeling.py)

```python
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
```

## Grocery and Gourmet Food
- [Source of data](http://jmcauley.ucsd.edu/data/amazon/qa/)

### Download and prepare data

1. Download: `$ wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Grocery_and_Gourmet_Food.json.gz`
2. Prepare train and verify data `$ python3 prepare_GG_data.py`
2. Prepare train and verify data for seq2seq `$ python3 prepare_seq_data.py`

#### qa_Grocery_and_Gourmet_Food Data format
Each line has a json object with the following properties:
- questionType
- asin
- answerTime
- unixTime
- question 
- answer

##### Sample code to load files
```python
import pandas as pd
import gzip

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

df = getDF('qa_Video_Games.json.gz')
```

## Models

### Text2Text-generations

- [hiiamsid/est5-base](https://huggingface.co/hiiamsid/est5-base?text=Tengo+un+problema+con+el+sistema+de+archivos%2C+parace+que+he+perdido+los+persmisos+para+acceder+a+los+ficheros+A3+y+A4.+¿Puedo+hacer+una+copia+de+seguridad+de+los+ficheros+A3+y+A4%3F)
- [mrm8488/spanish-t5-small-sqac-for-qa](https://huggingface.co/mrm8488/spanish-t5-small-sqac-for-qa?text=question%3A+¿Cuál+es+el+nombre+que+se+le+da+a+la+unidad+morfológica+y+funcional+de+los+seres+vivos%3F+context%3A+La+célula+%28del+lat%C3%ADn+cellula%2C+diminutivo+de+cella%2C+‘celda’%29+es+la+unidad+morfológica+y+funcional+de+todo+ser+vivo.+De+hecho%2C+la+célula+es+el+elemento+de+menor+tamaño+que+puede+considerarse+vivo.%E2%80%8B+De+este+modo%2C+puede+clasificarse+a+los+organismos+vivos+según+el+número+de+células+que+posean%3A+si+solo+tienen+una%2C+se+les+denomina+unicelulares+%28como+pueden+ser+los+protozoos+o+las+bacterias%2C+organismos+microscópicos%29%3B+si+poseen+más%2C+se+les+llama+pluricelulares.+En+estos+últimos+el+número+de+células+es+variable%3A+de+unos+pocos+cientos%2C+como+en+algunos+nematodos%2C+a+cientos+de+billones+%281014%29%2C+como+en+el+caso+del+ser+humano.+Las+células+suelen+poseer+un+tamaño+de+10+µm+y+una+masa+de+1+ng%2C+si+bien+existen+células+mucho+mayores.)

## Investigation ways
I must investigate four ways to solve this problem:
1. Train a lenguage model from scratch, to generate text.
2. Use a pre-trained language model retrained, to generate text. Like GPT-2.
3. Fine-tune a pre-trained language model, to generate text, by doing a "conversational" tasks.
4. Fine-tune a seq2seq model, to generate text.
5. Train from scratch a tiny model to get a 100% accuracy
6. Train a pytorch model based on translation
7. Use doc2vec to generate the question embedding, store it and find it using cosine similarity.
8. Try OpenAI to make a "text search" solution.
9. Try Huggingface BERT to make a "text search" solution.
10. Try Sentence-Transformers to make a "text search" solution.
11. Try NER.

### 1. Train a lenguage model from scratch, to generate text.
First install requirements:

    $ pip3 install -r ./transformers/examples/pytorch/language-modeling/requirements.txt

Also, do the step **Download and prepare data**.

Then run training:

    $ ./train_gpt2_from_scratch/train_gpt2_model.sh

After training is done, do inference:

    $ python3 ./train_gpt2_from_scratch/inference_gpt2_model.py

#### Results

Almost returns the same anwsers, no matter the questions: almos always returns: "I don't know"

### 2. Use a pre-trained language model retrained, to generate text. Like GPT-2.
First install requirements:

    $ pip3 install -r ./transformers/examples/pytorch/language-modeling/requirements.txt

Also, do the step **Download and prepare data**.

Then run training:

    $ ./train_gpt2_from_pretrained/train_gpt2_model.sh

After training is done, do inference:

    $ python3 ./train_gpt2_from_pretrained/inference_gpt2_model.py

#### Results

Training was fast, but i could not solve inference, because the script launch an exception that I could no solve.

### 3. Fine-tune a pre-trained language model, to generate text, by doing a "conversational" tasks
NoTHINMG

### 4. Fine-tune a seq2seq model, to generate text
NOTHOING

### 5. Train from scratch a tiny model to get a 100% accuracy
First install requirements:

    $ pip3 install -r ./transformers/examples/pytorch/language-modeling/requirements.txt

Then run training:

    $ ./train_tiny_gpt2_from_scratch/train_gpt2_model.sh

After training is done, do inference:

    $ python3 ./train_tiny_gpt2_from_scratch/inference_gpt2_model.py

### 6. Train a pytorch model based on translation
Use a translation model by adapting it to solve this questions-answering.

https://pytorch.org/tutorials/beginner/translation_transformer.html

ToDo:
- [X] Buscar códigos donde ya haya implementado Datasets.
- [X] Buscar códigos donde ya haya usado los transformers de pytorch.
- [X] Implementar el código tal y como está en el doc translation_transformer.
- [ ] Adaptar el código para hacer una question-answering.

tokenizer = get_tokenizer('basic_english')

- https://gitlab/ai/pytorch-word2vec/-/blob/main/word2vec/dataset.py
- https://gitlab/ai/libtorch-lm/-/blob/master/language_translation.py
- https://gitlab/ai/libtorch-lm/-/wikis/home
- https://andrewpeng.dev/transformer-pytorch/

Doc:
tokens
- UNK_IDX -> default index. This index is returned when the token is not found.
- PAD_IDX -> value used to fill short sequences.
- BOS_IDX -> begining of string
- EOS_IDX -> end of string
- SEP_IDX -> separator between questions and answers

Run sample file:    

    $ `source .venv/bin/activate`
    # Create source and target language tokenizer. Make sure to install the dependencies.
    # pip3 install -U spacy
    # python3 -m spacy download en_core_web_sm
    # python3 -m spacy download de_core_news_sm
    $ python3 ./train_pytorch_for_translation/language_translation.py

### 7. Use doc2vec to generate the question embedding, store it and find it using cosine similarity.
cosine-similarity(V,W) = 1 - (v1 * w1 + v2 * w2 + v3 * w3) / (sqrt(v1 * v1 + v2 * v2 + v3 * v3) * sqrt(w1 * w1 + w2 * w2 + w3 * w3)
)

Use StackOverflow dataset as corpus.

### 8. Try OpenAI to make a "text search" solution.
- [OpenAI: Embeddings](https://beta.openai.com/docs/guides/embeddings/)
- [Sqlite3 ruby gem doc](https://github.com/sparklemotion/sqlite3-ruby)
- [Wikipedia: Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Daru Gem (Pandas ruby alternative)](https://github.com/SciRuby/daru)
- [Daru doc form Ankane](https://ankane.org/daru)

- Use **Ada** model, because is the ligther (1024 dimensions)
- Use **text-similarity-ada-001** model for clustering, regression, anomaly detection, visualization.

#### Installation

##### First install last version of sqlite3
... deactivating **SQLITE_MAX_EXPR_DEPTH**

1. `CFLAGS="-DSQLITE_MAX_EXPR_DEPTH=0" ./configure` 
2. `make`
3. `sudo make install`

##### Set $PATH to point to /usr/local/bin 

`$ export PATH="$PATH:/usr/local/bin"`

#### Last install gems

`$ bundle`

**IMPORTANT** Install gems last to make sqlite3 gem use the compiled version of sqlite3 executable. 

#### Usage
First generate embeddings and store it in `sqlite.db`:

```sh
ruby ./openai_embeddings/generate_embeddings.rb
```

Then search for text:

```sh
ruby ./openai_embedding/search_text.rb
```


Use `train_tiny_list.csv`

#### Train

    $ python3 ./train_pytorch_for_translation/question-answering.py

### 9. Try Huggingface Bert to make a "text search" solution
with sentence embeddings

```sh
cd bert_embeddings
source .venv/bin/activate
python3 try_embeddings.py
```

###  10. Try Sentence-Transformers to make a "text search" solution
- [SentenceTransformers Documentation](https://www.sbert.net)
- [github(UKPLab/sentence-transformers)](https://github.com/UKPLab/sentence-transformers)
- [Pretrained models](https://www.sbert.net/docs/pretrained_models.html)

#### Use cases
- [Semantic Textual Similarity](https://www.sbert.net/docs/usage/semantic_textual_similarity.html)
- [Semantic search](https://www.sbert.net/examples/applications/semantic-search/README.html)
- [Paraphrase Mining](https://www.sbert.net/examples/applications/paraphrase-mining/README.html)

#### ToDo 
- [ ] See other use cases and usages of [Sentence-Transformers](https://www.sbert.net/examples/applications/computing-embeddings/README.html)

#### Install
First install PyTorch with CUDA.
Then:

    $ pip3 install -U sentence-transformers

#### Usage
```sh
cd sentence_transformers
source .venv/bin/activate
python3 try_sentence.py
```

#### Conclusions
Para idioma español funciona bastante bien el modelo:

Estos otros modelos también son multi-idioma:

- distiluse-base-multilingual-cased-v1
- distiluse-base-multilingual-cased-v2
- paraphrase-multilingual-MiniLM-L12-v2
- paraphrase-multilingual-mpnet-base-v2

### 11. Try NER
- [Huggingface: token classification](https://huggingface.co/tasks/token-classification)
- Davlan/bert-base-multilingual-cased-ner-hrl

```sh
.venv/bin/activate
cd ner
python3 try_spanish_ner.py
```

## ToDo
- [ ] Encontrar cómo guardar y restaurar modelos reentrenados.
- [X] Por lo que veo en el documento de custom dataset, lo que tengo que hacer es crear mi propio código que cargue los datos.
- [X] Investigar [Question Answering with SQuAD 2.0](https://huggingface.co/transformers/v3.2.0/custom_datasets.html#qa-squad)
- [X] Necesito encontrar la manera de reentrenar sistemas de **text2text-generation**, -> USAR LM https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling
- [X] Crear un sistema sencillo de `inference` de pipelines para **text2text-generation**.
- [X] Una opción que debo tener en cuenta es que quizás no necesito ralizar un fine-tuning, podría simplemente entrenar todo el modelo con mis correos. SEPARANDO LA PREGUNTA DE LA RESPUESTA CON UNA PALABRA TOKEN CLAVE COMO '>>>QA>>>'
- [X] Igual puedo hacer un reentreno de un gpt2 en español. ASÍ ES.

## Remember
- Text generation is currently possible with GPT-2, OpenAi-GPT, CTRL, XLNet, Transfo-XL and Reformer in PyTorch
- Una posible solución sería el **Text Generation**
- También se puede usar los transformers para el **Names Entity Recognition** (NER)
- Esto es lo que quiero hacer: fine-tuning un GPT-2 ---->>>> https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling#gpt-2gpt-and-causal-language-modeling

# Patches
## token patches
line run_clm.py:346

tokenizer.add_special_tokens({
        "eos_token": "</S>",
        "bos_token": "<S>",
        "unk_token": "<SEP>"
    })
    
# Marketing

- https://mailytica.com/en/pricing/
- https://emailtree.ai