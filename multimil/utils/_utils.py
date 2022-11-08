import pandas as pd
import torch

def create_df(pred, columns=None, index=None):

    if isinstance(pred, dict):
        for key in pred.keys():
            pred[key] = torch.cat(pred[key]).squeeze().cpu().numpy()
    else:
        pred = torch.cat(pred).squeeze().cpu().numpy()

    df = pd.DataFrame(pred)
    if index is not None:
        df.index = index
    if columns is not None:
        df.columns = columns
    return df