import numpy as np


# ===============================
# Membership Function From Scratch
# ===============================

def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif x == b:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)


def left_shoulder(x, a, b):
    if x <= a:
        return 1.0
    elif x >= b:
        return 0.0
    else:
        return (b - x) / (b - a)


def right_shoulder(x, a, b):
    if x <= a:
        return 0.0
    elif x >= b:
        return 1.0
    else:
        return (x - a) / (b - a)


# ===============================
# Fuzzification
# ===============================

def fuzzify_bmi(bmi):
    return {
        "normal": left_shoulder(bmi, 24, 27),
        "overweight": triangular(bmi, 24, 29, 34),
        "obese": right_shoulder(bmi, 30, 35)
    }


def fuzzify_age(age):
    return {
        "young": left_shoulder(age, 4, 6),
        "adult": triangular(age, 5, 8, 10),
        "old": right_shoulder(age, 9, 11)
    }


def fuzzify_genhlth(genhlth):
    return {
        "good": left_shoulder(genhlth, 2, 3),
        "fair": triangular(genhlth, 2, 3, 4),
        "poor": right_shoulder(genhlth, 3, 4)
    }


def fuzzify_binary(value):
    value = int(value)
    return {
        "no": 1.0 if value == 0 else 0.0,
        "yes": 1.0 if value == 1 else 0.0
    }


def fuzzify_activity(value):
    value = int(value)
    return {
        "inactive": 1.0 if value == 0 else 0.0,
        "active": 1.0 if value == 1 else 0.0
    }


def fuzzify_input(row):
    return {
        "BMI": fuzzify_bmi(row["BMI"]),
        "Age": fuzzify_age(row["Age"]),
        "GenHlth": fuzzify_genhlth(row["GenHlth"]),
        "HighBP": fuzzify_binary(row["HighBP"]),
        "HighChol": fuzzify_binary(row["HighChol"]),
        "PhysActivity": fuzzify_activity(row["PhysActivity"])
    }


# ===============================
# Rule Base
# ===============================

def apply_rules(fz):
    rules = []

    rules.append(("low", min(fz["BMI"]["normal"], fz["HighBP"]["no"], fz["HighChol"]["no"], fz["PhysActivity"]["active"])))
    rules.append(("low", min(fz["BMI"]["normal"], fz["GenHlth"]["good"], fz["Age"]["young"])))
    rules.append(("medium", min(fz["BMI"]["overweight"], fz["HighBP"]["no"], fz["HighChol"]["no"])))
    rules.append(("medium", min(fz["BMI"]["overweight"], fz["HighBP"]["yes"])))
    rules.append(("medium", min(fz["BMI"]["overweight"], fz["HighChol"]["yes"])))
    rules.append(("medium", min(fz["BMI"]["obese"], fz["HighBP"]["no"], fz["HighChol"]["no"])))
    rules.append(("high", min(fz["BMI"]["obese"], fz["HighBP"]["yes"])))
    rules.append(("high", min(fz["BMI"]["obese"], fz["HighChol"]["yes"])))
    rules.append(("high", min(fz["HighBP"]["yes"], fz["HighChol"]["yes"])))
    rules.append(("high", min(fz["GenHlth"]["poor"], fz["BMI"]["overweight"])))
    rules.append(("very_high", min(fz["GenHlth"]["poor"], fz["BMI"]["obese"])))
    rules.append(("high", min(fz["Age"]["old"], fz["HighBP"]["yes"])))
    rules.append(("high", min(fz["Age"]["old"], fz["HighChol"]["yes"])))
    rules.append(("high", min(fz["PhysActivity"]["inactive"], fz["BMI"]["obese"])))
    rules.append(("very_high", min(fz["PhysActivity"]["inactive"], fz["HighBP"]["yes"], fz["HighChol"]["yes"])))
    rules.append(("very_high", min(fz["Age"]["old"], fz["GenHlth"]["poor"], fz["BMI"]["obese"])))
    rules.append(("low", min(fz["GenHlth"]["good"], fz["PhysActivity"]["active"], fz["BMI"]["normal"])))
    rules.append(("medium", min(fz["Age"]["adult"], fz["BMI"]["overweight"], fz["HighBP"]["yes"])))

    return rules


# ===============================
# Mamdani
# ===============================

x_risk = np.arange(0, 101, 1)


def risk_membership(label, x):
    if label == "low":
        return left_shoulder(x, 25, 40)
    elif label == "medium":
        return triangular(x, 30, 50, 70)
    elif label == "high":
        return triangular(x, 60, 75, 90)
    elif label == "very_high":
        return right_shoulder(x, 80, 90)
    else:
        return 0.0


def mamdani_aggregation(rules):
    aggregated = []

    for x in x_risk:
        clipped_values = []

        for label, firing_strength in rules:
            firing_strength = float(firing_strength)
            mu_output = risk_membership(label, x)
            clipped_value = min(firing_strength, mu_output)
            clipped_values.append(clipped_value)

        aggregated_value = max(clipped_values) if clipped_values else 0.0
        aggregated.append(aggregated_value)

    return np.array(aggregated)


def mamdani_defuzzification(rules):
    aggregated = mamdani_aggregation(rules)

    numerator = np.sum(x_risk * aggregated)
    denominator = np.sum(aggregated)

    if denominator == 0:
        return 0.0

    crisp_output = numerator / denominator
    return crisp_output


def predict_mamdani(row):
    fz = fuzzify_input(row)
    rules = apply_rules(fz)
    risk_score = mamdani_defuzzification(rules)

    return risk_score


# ===============================
# Sugeno
# ===============================

sugeno_output = {
    "low": 20,
    "medium": 50,
    "high": 80,
    "very_high": 95
}


def sugeno_defuzzification(rules):
    numerator = 0.0
    denominator = 0.0

    for label, firing_strength in rules:
        firing_strength = float(firing_strength)
        z = sugeno_output[label]

        numerator += firing_strength * z
        denominator += firing_strength

    if denominator == 0:
        return 0.0

    crisp_output = numerator / denominator
    return crisp_output


def predict_sugeno(row):
    fz = fuzzify_input(row)
    rules = apply_rules(fz)
    risk_score = sugeno_defuzzification(rules)

    return risk_score


# ===============================
# Utility
# ===============================

def classify_risk(score, threshold=50):
    if score >= threshold:
        return 1
    else:
        return 0


def class_label(value):
    if int(value) == 0:
        return "Non-Diabetes"
    else:
        return "Diabetes"


def risk_category(score):
    if score < 40:
        return "Low Risk"
    elif score < 60:
        return "Medium Risk"
    elif score < 80:
        return "High Risk"
    else:
        return "Very High Risk"