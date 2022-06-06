from transformers import pipeline

qa_model = pipeline("text-generation", model="output_pretrained_wiki")

answer = qa_model("The game began development in 2010 , carrying over a large ", max_length=64)
print(answer)  # The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II . While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers . Character designer Raita Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa . A large team of writers handled the script . The game 's opening theme was sung by May 'n .
