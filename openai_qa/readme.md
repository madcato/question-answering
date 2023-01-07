# OpenAI Q&A
In this directory, I'm going to investigate how to solve Q&A from a custom plain text by using OpenAI models and APIs, also using fine-tuning and embeddings.

## Run basic usage
- `ruby test.rb` # run basic usage to learn how to send requests to OpenAI API

## Run preprocessing of a document library sample code
- `ruby preprocess.rb` # run preprocessing of a document library sample code


## More info
- [Fine-tune with OpenAI](https://beta.openai.com/docs/api-reference/fine-tunes)
- [Weaviate: vector database](https://github.com/semi-technologies/weaviate)
- [Qdrant: vector database](https://github.com/qdrant/qdrant)

### Posible Solución
La solución puede ser usar el davinci-instruct para generar preguntas y respuestas 

- [fine-tune QA, old](https://github.com/openai/openai-cookbook/blob/main/examples/fine-tuned_qa/olympics-2-create-qa.ipynb)

También tengo un comprobar el generar embeddings the textos, porque también tiene logo de generar preguntas:

- [QA using embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb)

### Email

- [Introducing text and code embeddings](https://openai.com/blog/introducing-text-and-code-embeddings/)

> We’re excited to announce a new embedding model: text-embedding-ada-002. This model unifies our 5 previous best-performing embedding models and is available today through our /embeddings API.
> 
> Embeddings measure the relatedness of text strings, which is useful for semantic search, cluster analysis, and other applications.
> 
> Our new embedding model is: 
> 	•	Better: it outperforms prior OpenAI models on most benchmark tasks.
> 	•	Simpler: a single model for both search and similarity tasks across both text and code.
> 	•	Able to read 4x more: it can embed up to 8,191 tokens (roughly ~10 pages) vs. 2,046 previously.
> 	•	10x more cost-effective: at $0.0004 / 1k tokens (or roughly ~3,000 pages per US dollar), it’s 10% the price of our previously lowest-priced embedding model.
> 
> For more information, see our blog post. We look forward to seeing what you build!
> 
> Thanks,
> The OpenAI team
> 

### Q&A sample
> 
> import os
> import openai
> 
> openai.api_key = os.getenv("OPENAI_API_KEY")
> 
> response = openai.Completion.create(
>   model="text-davinci-003",
>   prompt="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ: Where is the Valley of Kings?\nA:",
>   temperature=0,
>   max_tokens=100,
>   top_p=1,
>   frequency_penalty=0.0,
>   presence_penalty=0.0,
>   stop=["\n"]
> )