def Roberta_embeddings(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output
