from app.db.database import engine, Base
from app.db import models  # noqa: F401


def init_db():
    """Create all tables in the database. Fails gracefully if DB is unreachable."""
    try:
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully.")
    except Exception as e:
        print(f"Warning: Could not connect to database on startup: {e}")
        print("The app will start but database operations will fail until connection is restored.")


if __name__ == "__main__":
    init_db()