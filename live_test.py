import requests

if __name__ == "__main__":
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 20885,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = requests.post("https://udacity-census-ms.herokuapp.com/predict", json=payload)
    print(response.status_code)
    print(response.json())
