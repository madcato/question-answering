from transformers import pipeline

qa_model = pipeline("text-generation", model="output_tiny")
answer = qa_model("<S> Do they taste good? <SEP>", max_length=64)
print(answer)  # <S> Do they taste good? <SEP> Uhhh- don't know. They were super cute & just went into a gift bag. Never tasted them. </S>

answer = qa_model("<S> Are Larabars considered a raw food? <SEP>", max_length=64)
print(answer)  # <S> Are Larabars considered a "raw" food? <SEP> Yes, they are. I don't know if all of them are, but the ones I've enjoyed are raw. Visit the Larabar Website and see the "Raw Diet" portion. http://www.larabar.com/about/special-diets </S>

answer = qa_model("<S> Expedited ONLY? Are these dry or fresh noodles? <SEP>", max_length=64)
print(answer)  # <S> Expedited ONLY? Are these dry or fresh noodles? <SEP> I think they need to be refrigerated. </S>

answer = qa_model("<S> Is it dairy-free? <SEP>", max_length=64)
print(answer)
