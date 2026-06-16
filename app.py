from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templates import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

# Initialize the FastAPI app
app = FastAPI(title = "T5 Text Generation APP", description = "Text Summarization using T5 model", version = "1.0")

# Model and tokenizer initialization
model= T5ForConditionalGeneration.from_pretrained('/content/drive/MyDrive/Datasets/Text_Summarizer/saved_summary_model')
tokenizer =T5Tokenizer.from_pretrained('/content/drive/MyDrive/Datasets/Text_Summarizer/saved_summary_model')

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif  torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)