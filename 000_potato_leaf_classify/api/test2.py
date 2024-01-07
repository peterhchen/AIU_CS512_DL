import uvicorn

from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Hello test2"

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)