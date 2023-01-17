from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')                        # Embedding size: 768, este parece el mejor a día 20230117
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')  # Embedding size: 384
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v1')                       # Embedding size: 384
# model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')        # Embedding size: 768
# model = SentenceTransformer('Genario/multilingual_paraphrase')                              # Embedding size: 384
# model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')   # Embedding size: 512
# model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')                # Embedding size: 768
# model = SentenceTransformer('sentence-transformers/quora-distilbert-multilingual')          # Embedding size: 768
# model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')      # Embedding size: 512
# model = SentenceTransformer('sentence-transformers/use-cmlm-multilingual')                  # Embedding size: 768
# model = SentenceTransformer('sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-lng-aligned')  # Embedding size: 768
# model = SentenceTransformer('sentence-transformers/msmarco-distilbert-multilingual-en-de-v2-tmp-trained-scratch')  # Embedding size: 768
# model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli')  # en-es # Embedding size: 768  # Really bad
# model = SentenceTransformer('clips/mfaq')  # en-es # Embedding size: 768  # Really bad

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("Embedding vector size:", len(embedding))
    print("")

sentences = ["Hay que poner 34gr de azucar",
             "Poner 26gr de edulcorante en polvo",
             "Ir a pescar en verano es una buena idea",
             "No hay que poner 34gr de azucar",]
#Compute embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute cosine-similarities for each sentence with each other sentence
cosine_scores = util.cos_sim(embeddings, embeddings)

print(cosine_scores)
