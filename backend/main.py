from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers.cases import router as cases_router
from app.db.init_db import init_db

app = FastAPI(
    title="ImplantSAC API",
    description="Automated dental implant SAC classification pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases_router, prefix="/api/cases")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ImplantSAC API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}