
from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello test1"
