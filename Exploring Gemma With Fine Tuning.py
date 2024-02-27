# Databricks notebook source
import os
from dotenv import load_dotenv
import transformers
import torch
from datasets import load_dataset
from google.colab import userdata
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GemmaTokenizer

load_dotenv()

# COMMAND ----------

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
model_name = 'google/gemma-2b'

# COMMAND ----------


