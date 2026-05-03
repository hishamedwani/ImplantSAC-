import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Boolean, DateTime, JSON
from app.db.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Case(Base):
    __tablename__ = "cases"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, nullable=False, default="anonymous")
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Spacing
    spacing_x = Column(Float, nullable=False)
    spacing_y = Column(Float, nullable=False)
    spacing_z = Column(Float, nullable=False)

    # Measurements
    apical_bone_mm = Column(Float, nullable=False)
    buccal_wall_mm = Column(Float, nullable=False)
    ridge_width_mm = Column(Float, nullable=False)
    septum_width_mm = Column(Float, nullable=True)
    lesion_detected = Column(Boolean, nullable=False)
    lesion_size_mm3 = Column(Float, nullable=False)

    # Risk levels
    apical_risk = Column(String, nullable=False)
    buccal_risk = Column(String, nullable=False)
    ridge_risk = Column(String, nullable=False)
    septum_risk = Column(String, nullable=False)
    lesion_risk = Column(String, nullable=False)

    # Final classification
    classification = Column(String, nullable=False)
    reasoning = Column(JSON, nullable=False)

    # Full result JSON
    full_result = Column(JSON, nullable=False)