import data_processing as dp
import classifiers as clf
import geopandas as gpd
import pandas as pd
import numpy as np

train_data = gpd.read_file('../data/train.geojson')
test_data = gpd.read_file('../data/test.geojson')
train_data = train_data.set_index('index')
test_data = test_data.set_index('index')

pipeline = dp.Pipeline(
    dp.RemoveNones(test_data.columns),
    dp.SortDates(),
    dp.DifferencesInDates(),
    dp.OneHotEncoding(['change_status_date0','change_status_date1','change_status_date2','change_status_date3', 'change_status_date4']),
    dp.Replace('urban_type', {'N,A': 'N/A'}),
    dp.Replace('geography_type', {'N,A': 'N/A'}),
    dp.OneHotEncodingMulticlass('urban_type'),
    dp.OneHotEncodingMulticlass('geography_type'),
    dp.GeometryFeatures(),
    dp.MapLabels()
)
processed = pipeline(train_data)

X_train = processed.drop(columns='change_type').to_numpy(np.float32)
y_train = pd.DataFrame(processed.change_type).to_numpy(np.float32).squeeze()

classifier = clf.NeuralNet(X_train.shape[1], [256,128,64,16], 6, training_epochs=5)

y_train_one_hot = np.zeros((X_train.shape[0], 6))
y_train_one_hot[range(len(y_train)), y_train.astype(int)] = 1

classifier.train(X_train, y_train_one_hot)