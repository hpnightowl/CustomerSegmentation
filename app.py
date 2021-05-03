from flask import Flask
import pandas as pd
import pickle

app = Flask(__name__)

@app.route("/")
def home():
  ps = pd.read_csv('testdata/datatest.csv')
  model = pickle.load(open("model/model.pkl", "rb"))
  kmeans_clusters = model.predict(ps)
  print(kmeans_clusters)
  return str(kmeans_clusters)

if __name__ == '__main__':
    app.run()
