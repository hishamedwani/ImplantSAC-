from typing import Optional


# ------------------------------------------------------------------
# Per-factor threshold classifiers
# Based on ITI SAC classification guidelines
# ------------------------------------------------------------------

def classify_apical_bone(apical_mm: float) -> str:
    """
    Green:  height >= 4mm AND width >= 2mm
    Yellow: height 2-4mm OR width 1-2mm
    Red:    height < 2mm OR width < 1mm
    We only measure height here. Width is ridge width.
    """
    if apical_mm >= 4.0:
        return "Green"
    elif apical_mm >= 2.0:
        return "Yellow"
    else:
        return "Red"


def classify_buccal_wall(buccal_mm: float) -> str:
    """
    Green:  >= 2.0mm
    Yellow: 1.0 - 2.0mm
    Red:    < 1.0mm or absent
    """
    if buccal_mm >= 2.0:
        return "Green"
    elif buccal_mm >= 1.0:
        return "Yellow"
    else:
        return "Red"


def classify_septum(septum_mm: Optional[float]) -> str:
    """
    Only applies to molars.
    Green:  >= 3mm
    Yellow: 2-3mm
    Red:    < 2mm or absent
    """
    if septum_mm is None:
        return "N/A"
    if septum_mm >= 3.0:
        return "Green"
    elif septum_mm >= 2.0:
        return "Yellow"
    else:
        return "Red"


def classify_ridge_width(ridge_mm: float) -> str:
    """
    Green:  >= 7mm
    Yellow: 5-7mm
    Red:    < 5mm
    """
    if ridge_mm >= 7.0:
        return "Green"
    elif ridge_mm >= 5.0:
        return "Yellow"
    else:
        return "Red"


def classify_lesion(lesion_detected: bool, lesion_size_mm3: float) -> str:
    """
    Green:  No lesion
    Yellow: Lesion <= 3mm, no cortical perforation
    Red:    Lesion > 3mm or cortical perforation suspected
    We use lesion_size_mm3 as proxy. Cortical perforation
    requires radiologist confirmation — flagged as Yellow if
    lesion present but small.
    """
    if not lesion_detected:
        return "Green"
    # Approximate 3mm sphere volume = 4/3 * pi * 1.5^3 ~ 14.1 mm3
    if lesion_size_mm3 <= 14.1:
        return "Yellow"
    else:
        return "Red"


# ------------------------------------------------------------------
# Final SAC classification
# Based on ITI guidelines:
# Any Red    -> Complex
# Any Yellow -> Advanced
# All Green  -> Straightforward (Simple)
# ------------------------------------------------------------------

def final_sac(risk_colors: list[str]) -> str:
    valid = [c for c in risk_colors if c != "N/A"]
    if "Red" in valid:
        return "Complex"
    elif "Yellow" in valid:
        return "Advanced"
    else:
        return "Straightforward"


# ------------------------------------------------------------------
# Main classifier — takes measurement dict, returns full result
# ------------------------------------------------------------------

def classify_sac(measurements: dict) -> dict:
    """
    Takes output from compute_measurements() and returns
    full SAC classification with per-factor risk levels
    and reasoning chain.

    Args:
        measurements: dict from compute_measurements()

    Returns:
        Full classification result with reasoning
    """
    apical_mm = measurements["apical_bone_mm"]
    buccal_mm = measurements["buccal_wall_mm"]
    ridge_mm = measurements["ridge_width_mm"]
    septum_mm = measurements.get("septum_width_mm")
    lesion_detected = measurements["lesion_detected"]
    lesion_size_mm3 = measurements["lesion_size_mm3"]

    # Per-factor risk levels
    apical_risk = classify_apical_bone(apical_mm)
    buccal_risk = classify_buccal_wall(buccal_mm)
    ridge_risk = classify_ridge_width(ridge_mm)
    septum_risk = classify_septum(septum_mm)
    lesion_risk = classify_lesion(lesion_detected, lesion_size_mm3)

    all_risks = [apical_risk, buccal_risk, ridge_risk, lesion_risk]
    if septum_risk != "N/A":
        all_risks.append(septum_risk)

    classification = final_sac(all_risks)

    # Reasoning chain — explains why each factor got its risk level
    reasoning = []
    reasoning.append(
        f"Apical Bone: {apical_mm}mm → {apical_risk} "
        f"(threshold: ≥4mm Green, 2-4mm Yellow, <2mm Red)"
    )
    reasoning.append(
        f"Buccal Wall: {buccal_mm}mm → {buccal_risk} "
        f"(threshold: ≥2mm Green, 1-2mm Yellow, <1mm Red)"
    )
    reasoning.append(
        f"Ridge Width: {ridge_mm}mm → {ridge_risk} "
        f"(threshold: ≥7mm Green, 5-7mm Yellow, <5mm Red)"
    )
    if septum_risk != "N/A":
        reasoning.append(
            f"Interradicular Septum: {septum_mm}mm → {septum_risk} "
            f"(threshold: ≥3mm Green, 2-3mm Yellow, <2mm Red)"
        )
    reasoning.append(
        f"Periapical Lesion: {'Present' if lesion_detected else 'Absent'} "
        f"({lesion_size_mm3}mm³) → {lesion_risk}"
    )
    reasoning.append(f"Final SAC Classification: {classification}")

    return {
        "factors": {
            "apical_bone": {
                "measurement_mm": apical_mm,
                "risk": apical_risk
            },
            "buccal_wall": {
                "measurement_mm": buccal_mm,
                "risk": buccal_risk
            },
            "ridge_width": {
                "measurement_mm": ridge_mm,
                "risk": ridge_risk
            },
            "septum_width": {
                "measurement_mm": septum_mm,
                "risk": septum_risk
            },
            "periapical_lesion": {
                "lesion_detected": lesion_detected,
                "lesion_size_mm3": lesion_size_mm3,
                "risk": lesion_risk
            }
        },
        "classification": classification,
        "reasoning": reasoning,
        "disclaimer": (
            "This classification is a clinical decision support tool. "
            "Final treatment decisions remain the responsibility "
            "of the treating clinician."
        )
    }
