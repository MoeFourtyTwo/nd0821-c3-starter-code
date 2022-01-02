### Note

Github Link: https://github.com/MoeFourtyTwo/nd0821-c3-starter-code

Heroku App: https://udacity-census-ms.herokuapp.com/

Heroku App Docs: https://udacity-census-ms.herokuapp.com/docs

### Note

I used a machine learning library that can automatically create a model on the training data without the provided pre-process function. This model automatically includes the One Hot encoding. Therefore, there is no separate encoder saved in DVC.

### Run locally

Install requirements:
```
pip install -r requirements.txt -r requirements-dev.txt -e .
```

Start FastApi with
```
uvicorn main:app --reload
```

Train a new model with 
```
python starter/train_model.py
```