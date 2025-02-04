import pandas as pd

def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    for c in output.columns:
        output = output.drop(output.index[output[c].isnull()])

    return output