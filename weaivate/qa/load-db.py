import numpy as np
import weaviate
import pandas as pd
import pickle
import time

start = time.time()

## IMPORTANT: first run ../create-doc-schema.py

client = weaviate.Client("http://localhost:8080")

df = pd.read_csv('https://cdn.openai.com/API/examples/data/olympics_sections_text.csv')
df = df.set_index(["title", "heading"])
print(f"{len(df)} rows in the data.")
df.sample(5)

# run each element of dataframe
for index, row in df.iterrows():
    client.data_object.create(
        { 
          "content": row['content']
        },
        "Document"
    )

end = time.time()
print(end - start)
