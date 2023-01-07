require 'dotenv'
Dotenv.load
require 'daru'
require "numo/narray"
require './openai_utils'

context = "The Greek national anthem, Hymn to Liberty, was played before the marathon victory ceremonies to link the Ancient Olympics to the Modern Olympics. President of the IOC Thomas Bach (for Women's marathon), Vice-President of the IOC Anita DeFrantz (for Men's marathon) and World Athletics President Lord Sebastian Coe presented the medals to:"

openai = OpenAI_API.new(DOC_EMBEDDINGS_MODEL, QUERY_EMBEDDINGS_MODEL)

prompt = "Answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say \"I don't know\"

Context:
#{context}

Q: What is the olympic motto?
A:"

answer = openai.generate_answer(prompt)

p answer