MODEL_PATH = {

    "ElectraLarge": "Deehan1866/Finetuned-electra-large", #first app: 64.70, second app: 76.55
    #"Electra": "google/electra-base-discriminator", #first app: 61, second app: 75.25  
    #"StructBert": "bayartsogt/structbert-large", #first app: 59, second app: 70.50
    #"ElectraSmall":"google/electra-small-discriminator", #first app: 58.25, second app: 70.20
    #"Distilbert": "distilbert/distilbert-base-uncased",  #GOOD, 64.50, second app: 69.55
    #"SpanBERT-large": "SpanBERT/spanbert-large-cased",
    #"multiqaMPnet": "sentence-transformers/multi-qa-mpnet-base-dot-v1", #63.75, second app: 67.25
    #"paraphraseMPnet": "sentence-transformers/paraphrase-mpnet-base-v2", #65.30, second app: 67.15
    #"DPRport": "sentence-transformers/facebook-dpr-ctx_encoder-multiset-base", #62.4, second app: 66.50
    #"mpNET": "sentence-transformers/nli-mpnet-base-v2", #63.36, second app: 66.50

    
    #"PhraseBERT": "whaleloops/phrase-bert",
    # "DensePhrases": "princeton-nlp/densephrases-multi-query-multi",
    #"BERT": "bert-base-uncased",
    #"SpanBERT-base": "SpanBERT/spanbert-base-cased",
    #"SpanBERT-large": "SpanBERT/spanbert-large-cased",
    #"SentenceBERT": "bert-base-nli-stsb-mean-tokens",
    #"SimCSE": "princeton-nlp/sup-simcse-bert-base-uncased",



    #RESULT TOO LOW
    
    #"DistilbertFinetuned": "distilbert/distilbert-base-uncased-finetuned-sst-2-english", #62, second app: 63.90
    #"E5": "intfloat/e5-base-v2", #first app: 61.05, second app: 61.55
    #"E5Large": "intfloat/e5-large-v2", #firs app: 62, second app: 62.40
    #"DistilbertQA": "distilbert/distilbert-base-cased-distilled-squad", #second app: 61.80
    #"sentenceDISTILBERT": "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking",#second app: 57.15
    #"Canine": "google/canine-c", #50 accuracy, second app: 46.30
    #"sbert": "l3cube-pune/indic-sentence-similarity-sbert", #58.50 accuracy, second app: 55.55
    #"miniLM": "sentence-transformers/all-MiniLM-L6-v2", #58 accuracy, second app: 54.50
    #"MiniLM": "sentence-transformers/all-MiniLM-L12-v2", #58, second app: 56.45
    #"Minilm": "sentence-transformers/paraphrase-MiniLM-L6-v2", #54.75, second app: 55.85
    #"MINILM": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1", #56.75, second app: 58.75
    #"Mpnet": "sentence-transformers/all-mpnet-base-v2", #64, second app: 62.45
    #"MpnetV1": "sentence-transformers/all-mpnet-base-v1", #65.10, second app: 63.75
    #"MpNET": "sentence-transformers/stsb-mpnet-base-v2", #62.95, second app: 64.85
    #"Labse": "sentence-transformers/LaBSE", #55, second app: 54.05
    #"SentenceDistillbert": "sentence-transformers/msmarco-distilbert-dot-v5" #56.56
    #"SentenceBert": "sentence-transformers/msmarco-bert-base-dot-v5" #56.40


    #TOO LARGE
    #"T511B": "google-t5/t5-11b", #too large
    #"LLM": "meta-llama/Meta-Llama-3-8B", #too big
    #"EmbedMistral": "Linq-AI-Research/Linq-Embed-Mistral", #torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU 
    #"EMistral": "intfloat/e5-mistral-7b-instruct",  #not working
    #"Mistral": "mistralai/Mistral-7B-v0.1", #same issue as the previous one
    #"Xlnet": "xlnet/xlnet-base-cased", #OverflowError: can't convert negative int to unsigned
    
    
    #Too MANY PHRASES NOT FOUND
    #"E5LargeMultilingual": "intfloat/multilingual-e5-large-instruct" #tooo many, still 61
    #"Albert": "albert/albert-base-v2" #many many phrases are not found! accuracy is really low
    #"alBERT": "sentence-transformers/paraphrase-albert-small-v2", #too many phrases, 55.73 
    #"DistilbertGPT": "google-t5/t5-large" #error not enough values to unpack
    #"Roberta": "FacebookAI/roberta-base" #too many phrase not found
    #"DistilRoberta": "distilbert/distilroberta-base", #first app: 50.00, too many phrases not found, second app:51.16
    #"XLMRoberta": "FacebookAI/xlm-roberta-base" #too many phrase not found bt still gives 58
    #"RobertaEnglish": "FacebookAI/xlm-roberta-large-finetuned-conll03-english" #too many phrase not found
    #"BigBird": "google/bigbird-roberta-base" #61 accuracy too many phrases not found
    #"BigBirdLarge": "google/bigbird-roberta-large" #59 too too too many phrases
    #"Longformer": "allenai/longformer-base-4096" #too too too too too too too too
    #"T5": "google-t5/t5-large" #tooo many, error dise last e (not many values to unpack)
    #"FlanT5": "google/flan-t5-base",
    #"Luke": "studio-ousia/luke-base" #too many, 55
    #"SentenceT5": "sentence-transformers/sentence-t5-base", #61.06, phrases not found, second app: 57.91
    #"SentenceSimilarity": "annakotarba/sentence-similarity" #too many, 55 accuracy
    #"SentenceROberta": "sentence-transformers/roberta-base-nli-stsb-mean-tokens", #68.60, so many phrases, second app: 20.93
    #"DPR": "facebook/dpr-question_encoder-single-nq-base" #Dimension issues
    #"Deberta": "microsoft/deberta-v3-base",
    
    
}