import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from tqdm import tqdm # Used for progress bar during walk-forward validation


