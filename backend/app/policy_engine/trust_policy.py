def select_policy(risk_score: float):
    """
    Policy selection based on AI risk score.
    """

    if risk_score >= 0.8:
        return "HIGH_RISK"
    elif risk_score >= 0.5:
        return "MEDIUM_RISK"
    else:
        return "LOW_RISK"
