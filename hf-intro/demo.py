from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1) 最高层：pipeline
clf = pipeline("sentiment-analysis")
print(clf(["I love this course!", "This is terrible."]))

# 2) 拆开 pipeline：手动 tokenizer -> model -> logits -> softmax
name = "distilbert-base-uncased-finetuned-sst-2-english"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name)
inputs = tok(["I love this course!", "This is terrible."],
             padding=True, truncation=True, return_tensors="pt")
print(inputs)                       # 看 input_ids / attention_mask
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.softmax(logits, dim=-1)
print(probs, model.config.id2label)  # 确认和 pipeline 输出一致