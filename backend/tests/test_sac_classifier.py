import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.classification.sac_classifier import classify_sac


def test_straightforward():
    measurements = {
        "apical_bone_mm": 6.0,
        "buccal_wall_mm": 2.5,
        "ridge_width_mm": 8.0,
        "septum_width_mm": 4.0,
        "lesion_detected": False,
        "lesion_size_mm3": 0.0
    }
    result = classify_sac(measurements)
    assert result["classification"] == "Straightforward"
    print("Straightforward case passed ✓")
    return result


def test_advanced():
    measurements = {
        "apical_bone_mm": 3.0,
        "buccal_wall_mm": 2.5,
        "ridge_width_mm": 8.0,
        "septum_width_mm": None,
        "lesion_detected": False,
        "lesion_size_mm3": 0.0
    }
    result = classify_sac(measurements)
    assert result["classification"] == "Advanced"
    print("Advanced case passed ✓")
    return result


def test_complex():
    measurements = {
        "apical_bone_mm": 1.0,
        "buccal_wall_mm": 0.5,
        "ridge_width_mm": 8.0,
        "septum_width_mm": None,
        "lesion_detected": True,
        "lesion_size_mm3": 20.0
    }
    result = classify_sac(measurements)
    assert result["classification"] == "Complex"
    print("Complex case passed ✓")
    return result


if __name__ == "__main__":
    print("\n--- Testing SAC Classifier ---\n")

    r1 = test_straightforward()
    print("Reasoning:")
    for line in r1["reasoning"]:
        print(f"  {line}")

    print()
    r2 = test_advanced()
    print("Reasoning:")
    for line in r2["reasoning"]:
        print(f"  {line}")

    print()
    r3 = test_complex()
    print("Reasoning:")
    for line in r3["reasoning"]:
        print(f"  {line}")

    print("\nAll tests passed ✓")
