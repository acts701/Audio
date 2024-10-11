from kss import split_sentences
import torch
from sentence_transformers import SentenceTransformer, util
import configparser

config = configparser.ConfigParser()
config.read('config.json')
downpath = config['PATH']['DOWNLOAD_FOLDER']

with open(config["PATH"]["AUDIO_SAMPLE_FILE"]) as file_path:
  contents = file_path.read()
  sentences = split_sentences(contents)  

model = SentenceTransformer("Huffon/sentence-klue-roberta-base")

document_embeddings = model.encode(sentences)

query = config["SampleQuery"]["ISAIAH01"]
query_embedding = model.encode(query)

top_k = min(5, len(sentences))
cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

print(f"입력 문장: {query}")
print(f"<입력 문장과 유사한 {top_k} 개의 문장>")

for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
    print(f"{i+1}: {sentences[idx]} {'(유사도: {:.4f})'.format(score)}")
