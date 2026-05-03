from app.db.database import engine, Base
from app.db import models  # noqa: F401 — ensures models are registered


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")


if __name__ == "__main__":
    init_db()