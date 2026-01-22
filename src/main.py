import pandas as pd

from features.user_features import build_user_features
from data.build_dataset import build_dataset
from model.train import train_model
from model.evaluate import simulate_campaign

# Chargement des données
orders = pd.read_csv("../data/orders.csv")
prior = pd.read_csv("../data/order_products__prior.csv")
train = pd.read_csv("../data/order_products__train.csv")

# Features client
user_features = build_user_features(orders)

# Dataset ML
dataset = build_dataset(train, prior, orders, user_features)

# Entraînement
model, auc = train_model(dataset)
print(f"AUC: {auc:.3f}")

# Simulation business
campaign_result = simulate_campaign(model, dataset)
print(campaign_result)
