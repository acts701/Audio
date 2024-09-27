from kss import split_sentences
import torch
from sentence_transformers import SentenceTransformer, util

with open(r'C:/sermons/Isaiah/160601_Isaiah01.txt') as file_path:
  contents = file_path.read()
  sentences = split_sentences(contents)  

model = SentenceTransformer("Huffon/sentence-klue-roberta-base")

document_embeddings = model.encode(sentences)

query = "이사야에 나온 출애굽기 용어는"
query_embedding = model.encode(query)

top_k = min(5, len(sentences))
cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
top_results = torch.topk(cos_scores, k=top_k)

print(f"입력 문장: {query}")
print(f"<입력 문장과 유사한 {top_k} 개의 문장>")

for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
    print(f"{i+1}: {sentences[idx]} {'(유사도: {:.4f})'.format(score)}")
