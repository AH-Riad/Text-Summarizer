from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templates import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

app = FastAPI()