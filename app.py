import os
import json
from typing import Dict, Any
import pdfplumber
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY=os.getenv("GOOGLE_API")
llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash",api_key=GOOGLE_API_KEY)

@tool
def placement_agent(inputs: Dict[str,Any])->Dict[str,Any]:
    return {"message":"Implement your placement workflow here.","inputs":inputs}

chain=RunnableLambda(lambda x: placement_agent.invoke(x))

app=FastAPI(title="Placement Agent")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])
add_routes(app,chain,path="/agent",playground_type="default")

@app.get("/")
def root():
    return {"message":"Visit /agent/playground"}

@app.post("/analyze")
async def analyze(resume:UploadFile=File(...),target_role:str=Form(...),github_username:str=Form(...)):
    temp=f"/tmp/{resume.filename}"
    with open(temp,"wb") as f:
        f.write(await resume.read())
    return placement_agent.invoke({
        "resume_pdf_path":temp,
        "target_role":target_role,
        "github_username":github_username
    })

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT",8000)))
