import os
from dotenv import load_dotenv
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GemmaTokenizer

load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
model_name = 'google/gemma-2b'

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            token=os.environ['HF_TOKEN'])

# generate some text
text = "Quote: Why we suffer "

inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))