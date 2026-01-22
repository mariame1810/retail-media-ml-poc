def simulate_campaign(model, dataset, quantile=0.9):
    features = ["nb_orders", "avg_days_between_orders"]

    dataset = dataset.copy()
    dataset["score"] = model.predict_proba(dataset[features])[:, 1]

    threshold = dataset["score"].quantile(quantile)
    dataset["targeted"] = (dataset["score"] >= threshold).astype(int)

    return dataset.groupby("targeted")["target"].mean()
