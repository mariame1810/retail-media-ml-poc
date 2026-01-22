import pandas as pd

def build_user_features(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Features client à partir des commandes.
    """
    user_features = (
        orders
        .groupby("user_id")
        .agg(
            nb_orders=("order_id", "nunique"),
            avg_days_between_orders=("days_since_prior_order", "mean")
        )
        .reset_index()
    )
    return user_features
