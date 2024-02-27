# Databricks notebook source
import os
from dotenv import load_dotenv
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GemmaTokenizer

load_dotenv()

# COMMAND ----------

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
model_name = 'google/gemma-2b'

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            token=os.environ['HF_TOKEN'])

# COMMAND ----------


