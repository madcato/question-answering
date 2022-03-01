from util.text import *

df = getDF('qa_Grocery_and_Gourmet_Food.json.gz')
print("DataFrame contents:");
print(df);
print("Summary of the DataFrame:");
print(df.info());

questions, answers = split_data(df)

# split data
from sklearn.model_selection import train_test_split
train_questions, val_questions, train_answers, val_answers = train_test_split(questions, answers, test_size=.2)

print("Question 1:")
print(train_questions[1])
print("Answer 1:")
print(train_answers[1])

# Tokenize
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings_questions = tokenizer(train_questions, truncation=True, padding=True)
train_encodings_answers = tokenizer(train_answers, truncation=True, padding=True)
val_encodings_questions = tokenizer(val_questions, truncation=True, padding=True)
val_encodings_answers = tokenizer(val_answers, truncation=True, padding=True)

print("Encoded Question 1:")
print(train_encodings_questions[0])
print("Encoded Answer 1:")
print(train_encodings_answers[1])

# dataset
from grocery_gourmet_dataset import GroceryGourmetDataset
train_dataset = GroceryGourmetDataset(train_encodings_questions, train_encodings_answers)
val_dataset = GroceryGourmetDataset(val_encodings_questions, val_encodings_answers)
