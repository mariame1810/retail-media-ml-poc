import pandas as pd

def build_dataset(train, prior, orders, user_features):
    """
    Construit un dataset ML avec positifs et négatifs.
    """
    train = train.copy()
    train["target"] = 1

    negatives = prior.sample(n=len(train), random_state=42)
    negatives = negatives.merge(
        orders[["order_id", "user_id"]],
        on="order_id",
        how="left"
    )
    negatives["target"] = 0

    negatives = negatives[train.columns]

    dataset = pd.concat([train, negatives], ignore_index=True)
    dataset = dataset.merge(user_features, on="user_id", how="left")
    dataset = dataset.fillna(0)

    return dataset
