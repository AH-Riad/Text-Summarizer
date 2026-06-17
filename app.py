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

# Set up Jinja2 templates
templates = Jinja2Templates(directory=("."))

# Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str
    
def clean_data(text):
    text = str(text)
    text = re.sub(r"\r\n", " ", text) #Lines
    text = re.sub(r"\s+", " ", text) #Spaces
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower()

    return text

def summarize_dialogue(dialogue:str) -> str:
  dialogue = clean_data(dialogue)   # clean

  # Tokenize
  inputs = tokenizer(
      dialogue,
      padding = "max_length",
      max_length = 512,
      truncation = True,
      return_tensors = "pt"
  ).to(device)

  # Generate the summary => token_ids
  model.to(device)

  targets = model.generate(
      input_ids = inputs['input_ids'],
      attention_mask = inputs['attention_mask'],
      max_length = 150,
      num_beams = 4,
      repetition_penalty = 2.5,
      length_penalty = 1.0,
      early_stopping = True
  )

  # decode our output

  summary = tokenizer.decode(targets[0], skip_special_tokens=True)

  return summary

# API endpoint for summarization
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse) 
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})