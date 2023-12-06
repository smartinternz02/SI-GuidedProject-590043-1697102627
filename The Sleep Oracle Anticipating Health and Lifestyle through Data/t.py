from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
model = pickle.load(open('xgbc.pkl', 'rb'))
print(model.predict([[1,20,3,0,8,23,4,0,80,32000,120,80]]))