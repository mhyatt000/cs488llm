import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    print("Hello from cs488llm-hw!")

    url_path = 'google/gemma-3-27b-it'
    url_path = "google/gemma-7b"
    url_path = "google/gemma-3-1b-it"

    tokenizer = AutoTokenizer.from_pretrained(url_path)
    model = AutoModelForCausalLM.from_pretrained(url_path, device_map="auto")

    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
