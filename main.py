import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ETA Backend API")

# CORS - allow Lovable frontend
origins = [
    "https://lovable.dev",
    "https://*.lovable.dev",
    "http://localhost:5173",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "ETA Backend API is running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/test")
def test_endpoint():
    return {"data": "Hello from ETA Backend!", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
