import pandas as pd

from fuzzy_logic import (
    classify_risk,
    predict_mamdani,
    predict_sugeno,
    risk_category,
)


def test_reference_sample_scores():
    sample = pd.Series(
        {
            "BMI": 26.0,
            "Age": 4.0,
            "GenHlth": 3.0,
            "HighBP": 1.0,
            "HighChol": 0.0,
            "PhysActivity": 1.0,
        }
    )

    assert predict_mamdani(sample) == 50.0
    assert predict_sugeno(sample) == 50.0


def test_low_risk_sample():
    sample = pd.Series(
        {
            "BMI": 18.0,
            "Age": 1.0,
            "GenHlth": 3.0,
            "HighBP": 0.0,
            "HighChol": 0.0,
            "PhysActivity": 1.0,
        }
    )

    mamdani_score = predict_mamdani(sample)
    sugeno_score = predict_sugeno(sample)

    assert round(mamdani_score, 6) == 16.282828
    assert sugeno_score == 20.0
    assert classify_risk(mamdani_score) == 0
    assert classify_risk(sugeno_score) == 0
    assert risk_category(mamdani_score) == "Low Risk"
    assert risk_category(sugeno_score) == "Low Risk"
