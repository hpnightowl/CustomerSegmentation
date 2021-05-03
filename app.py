from flask import Flask
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def home():
  ps = pd.read_csv('testdata/datatest.csv')
  model = pickle.load(open("model/model.pkl", "rb"))
  prediction = model.predict(ps)
  if float(prediction) == 0:
    cluster = 'BOGO offers'
  elif float(prediction) == 1:
    cluster = 'Receive a higher than average number of offers'
  elif float(prediction) == 2:
    cluster = 'no BOGO offers'
  elif float(prediction) == 3:
    cluster = 'Regular offers'
  return str(cluster)

if __name__ == '__main__':
    app.run()
