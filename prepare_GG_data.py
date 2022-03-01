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

train_pairs = list(zip(train_questions, train_answers))
val_pairs = list(zip(val_questions, val_answers))

train_list = list(map(qa_joiner, train_pairs))
val_list = list(map(qa_joiner, val_pairs))

print(train_list[1])

print("Saving train_list to train_list.txt")
save_list_to_file(train_list, "train_list.txt")
print("Saving val_list to val_list.txt")
save_list_to_file(val_list, "val_list.txt")
