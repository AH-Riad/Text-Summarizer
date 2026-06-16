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
