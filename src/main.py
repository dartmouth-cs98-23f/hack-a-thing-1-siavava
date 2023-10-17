#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi import FastAPI

from network import GPTLanguageModel, load_model, encode, decode, torch, device

app = FastAPI()
model = load_model("network/models/transfusion.pth")

@app.get("/")
async def root():
  return {"message": "Hello World"}

@app.get("/api/{query}")
async def api(query: str):

  # create torch tensor with query as 2D array
  context = torch.tensor( [encode(query)], dtype=torch.long, device=device)
  
  # generate from context
  generated = model.generate(context, max_new_tokens=150)
  
  # decode generated response
  response = decode(generated[0].tolist())

  return { "query": query, "response": response }
