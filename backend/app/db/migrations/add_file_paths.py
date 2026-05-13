from app.db.database import engine
from sqlalchemy import text


def migrate():
    """Add segmentation_path and cbct_path columns to cases table."""
    with engine.connect() as conn:
        try:
            conn.execute(text(
                "ALTER TABLE cases ADD COLUMN segmentation_path VARCHAR"
            ))
            print("Added segmentation_path column")
        except Exception as e:
            print(f"segmentation_path already exists or error: {e}")

        try:
            conn.execute(text(
                "ALTER TABLE cases ADD COLUMN cbct_path VARCHAR"
            ))
            print("Added cbct_path column")
        except Exception as e:
            print(f"cbct_path already exists or error: {e}")

        try:
            conn.execute(text(
                "ALTER TABLE cases ADD COLUMN yolo_z INTEGER"
            ))
            conn.execute(text(
                "ALTER TABLE cases ADD COLUMN yolo_cx INTEGER"
            ))
            conn.execute(text(
                "ALTER TABLE cases ADD COLUMN yolo_cy INTEGER"
            ))
            print("Added yolo columns")
        except Exception as e:
            print(f"yolo columns already exist or error: {e}")

        conn.commit()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()