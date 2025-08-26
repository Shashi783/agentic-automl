def verify_dummy_token(token: str):
    """
    Dummy token verifier.
    Accepts "student-token" or "creator-token".
    """
    if token == "student-token":
        return {"email": "student@example.com", "role": "student"}
    elif token == "creator-token":
        return {"email": "creator@example.com", "role": "creator"}
    return None