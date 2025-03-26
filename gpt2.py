from pathlib import Path

from transformers import GPT2Model, GPT2Tokenizer

path = Path().home().parent / "cs488llm/models/models--gpt2/"
print(path)

path = (
    Path().home().parent
    / "cs488llm/models/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
)

tokenizer = GPT2Tokenizer.from_pretrained(path)
model = GPT2Model.from_pretrained("gpt2")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)

print(output)
