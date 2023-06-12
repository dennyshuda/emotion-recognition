from transformers import pipeline

nlp = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

text = "Aku tidak tahu harus bagaimana lagi"

result = nlp(text)

for prediction in result:
  label = prediction['label']
  score = prediction['score']
  print(f"Emosi: {label}, Skor: {score}")
