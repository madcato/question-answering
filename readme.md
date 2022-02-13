# Question anwsering

Actually, for solving a **question-answering** problem like the email answering, we must use **text-generation** solutions, the type of task we must use **text2text-generation**. Like:

- [Text Generation](https://huggingface.co/tasks/text-generation)
- [Hugging Face: install from source](https://huggingface.co/docs/transformers/installation#installing-from-source)
- [transformers/examples/pytorch/text-generation/](https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-generation)
- [Hugging Face - Text2Text Generation models](https://huggingface.co/models?language=es&library=pytorch&pipeline_tag=text2text-generation&sort=downloads)
- [Huggingface: fine-tuning with custom datasets](https://huggingface.co/transformers/v3.2.0/custom_datasets.html)
- [transformers/examples/pytorch/language-modeling/](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)

## Models

### Text2Text-generations

- [hiiamsid/est5-base](https://huggingface.co/hiiamsid/est5-base?text=Tengo+un+problema+con+el+sistema+de+archivos%2C+parace+que+he+perdido+los+persmisos+para+acceder+a+los+ficheros+A3+y+A4.+¿Puedo+hacer+una+copia+de+seguridad+de+los+ficheros+A3+y+A4%3F)
- [mrm8488/spanish-t5-small-sqac-for-qa](https://huggingface.co/mrm8488/spanish-t5-small-sqac-for-qa?text=question%3A+¿Cuál+es+el+nombre+que+se+le+da+a+la+unidad+morfológica+y+funcional+de+los+seres+vivos%3F+context%3A+La+célula+%28del+lat%C3%ADn+cellula%2C+diminutivo+de+cella%2C+‘celda’%29+es+la+unidad+morfológica+y+funcional+de+todo+ser+vivo.+De+hecho%2C+la+célula+es+el+elemento+de+menor+tamaño+que+puede+considerarse+vivo.%E2%80%8B+De+este+modo%2C+puede+clasificarse+a+los+organismos+vivos+según+el+número+de+células+que+posean%3A+si+solo+tienen+una%2C+se+les+denomina+unicelulares+%28como+pueden+ser+los+protozoos+o+las+bacterias%2C+organismos+microscópicos%29%3B+si+poseen+más%2C+se+les+llama+pluricelulares.+En+estos+últimos+el+número+de+células+es+variable%3A+de+unos+pocos+cientos%2C+como+en+algunos+nematodos%2C+a+cientos+de+billones+%281014%29%2C+como+en+el+caso+del+ser+humano.+Las+células+suelen+poseer+un+tamaño+de+10+µm+y+una+masa+de+1+ng%2C+si+bien+existen+células+mucho+mayores.)
## ToDo
- [ ] Por lo que veo en el documento de custom dataset, lo que tengo que hacer es crear mi propio código que cargue los datos.
- [X] Investigar [Question Answering with SQuAD 2.0](https://huggingface.co/transformers/v3.2.0/custom_datasets.html#qa-squad)
- [ ] Necesito encontrar la manera de reentrenar sistemas de **text2text-generation**, 
- [ ] Crear un sistema sencillo de `inference` de pipelines para **text2text-generation**.
- [ ] Encontrar cómo guardar y restaurar modelos reentrenados.
- [ ] Una opción que debo tener en cuenta es que quizás no necesito ralizar un fine-tuning, podría simplemente entrenar todo el modelo con mis correos.
- [ ] Igual puedo hacer un reentreno de un gpt2 en español.