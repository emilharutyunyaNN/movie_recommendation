import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from typing import List
from tensorflow import keras
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from ncf import *
from utils import *


# Versions of libraries used

print(f"Tensorflow version: {tf.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")

TOP_K = 100
N_EPOCHS = 10
data = fetch_movielens()

# Looking into the data, before processing
print("Interaction matrix:")
print(data["train"].toarray()[:100, :100])

for dataset in ["test", "train"]:
    data[dataset] = (data[dataset].toarray() > 0).astype("int8")

    # Make the ratings binary
print("Interaction matrix:")
print(data["train"][:100, :100])

print("\nRatings:")
unique_ratings = np.unique(data["train"])
print(unique_ratings)
    


from keras.optimizers import Adam



long_train = wide_to_long(data["train"], unique_ratings)
df_train = pd.DataFrame(long_train, columns=["user_id", "item_id", "interaction"])

# hide_input
print("All interactions:")
df_train.head()
print("Only positive interactions:")
df_train[df_train["interaction"] > 0].head()


# Neural Colaborative Filtering NCF
   
n_users, n_items = data["train"].shape
ncf_model = create_ncf(n_users, n_items)

ncf_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)
ncf_model._name = "neural_collaborative_filtering"
ncf_model.summary()

# Making tensorflow dataset
ds_train, ds_val = make_tf_dataset(df_train, ["interaction"])


# define logs and callbacks
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=0
)

train_hist = ncf_model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=N_EPOCHS,
    callbacks=[tensorboard_callback, early_stopping_callback],
    verbose=1,
)



long_test = wide_to_long(data["train"], unique_ratings)
df_test = pd.DataFrame(long_test, columns=["user_id", "item_id", "interaction"])
ds_test, _ = make_tf_dataset(df_test, ["interaction"], val_split=0, seed=None)

ncf_predictions = ncf_model.predict(ds_test)
df_test["ncf_predictions"] = ncf_predictions

std = df_test.describe().loc["std", "ncf_predictions"]
if std < 0.01:
    raise ValueError("Model predictions have standard deviation of less than 1e-2.")


data["ncf_predictions"] = df_test.pivot(
    index="user_id", columns="item_id", values="ncf_predictions"
).values
print("Neural collaborative filtering predictions")
print(data["ncf_predictions"][:100, :4])


precision_ncf = tf.keras.metrics.Precision(top_k=TOP_K)
recall_ncf = tf.keras.metrics.Recall(top_k=TOP_K)
accuracy_ncf = tf.keras.metrics.TopKCategoricalAccuracy(k=TOP_K) 
#auc_ncf = tf.keras.metrics.AUC(k=TOP_K) 
precision_ncf.update_state(data["test"], data["ncf_predictions"])
recall_ncf.update_state(data["test"], data["ncf_predictions"])
accuracy_ncf.update_state(data["test"], data["ncf_predictions"])
print(
    f"At K = {TOP_K}, we have a precision of {precision_ncf.result().numpy():.5f}",
    f"and a recall of {recall_ncf.result().numpy():.5f}",
    f"and a accuracy of {accuracy_ncf.result().numpy():.5f}"
)


lightfm_model = LightFM(loss="warp")
lightfm_model.fit(sparse.coo_matrix(data["train"]), epochs=N_EPOCHS)

lightfm_predictions = lightfm_model.predict(
    df_test["user_id"].values, df_test["item_id"].values
)
df_test["lightfm_predictions"] = lightfm_predictions
wide_predictions = df_test.pivot(
    index="user_id", columns="item_id", values="lightfm_predictions"
).values
data["lightfm_predictions"] = norm(wide_predictions)

# compute the metrics
precision_lightfm = tf.keras.metrics.Precision(top_k=TOP_K)
recall_lightfm = tf.keras.metrics.Recall(top_k=TOP_K)
accuracy_lightfm = tf.keras.metrics.TopKCategoricalAccuracy(k=TOP_K)
precision_lightfm.update_state(data["test"], data["lightfm_predictions"])
recall_lightfm.update_state(data["test"], data["lightfm_predictions"])
accuracy_lightfm.update_state(data["test"], data["lightfm_predictions"])
print(
    f"At K = {TOP_K}, we have a precision of {precision_lightfm.result().numpy():.5f}",
    f"and a recall of {recall_lightfm.result().numpy():.5f}",
    f"and a accuracy of {accuracy_lightfm.result().numpy():.5f}"
)


##### PLOTS ######################
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# ...
# Define the ground truth
y_true = data["test"].ravel()

# NCF predictions
y_pred_ncf = data["ncf_predictions"].ravel()

# LightFM predictions
y_pred_lightfm = data["lightfm_predictions"].ravel()

# Convert predictions to binary (assuming a threshold of 0.5)
y_pred_ncf_binary = (y_pred_ncf > 0.5).astype(int)
y_pred_lightfm_binary = (y_pred_lightfm > 0.5).astype(int)

# Compute and print accuracy, precision, recall, and F1 score


print_metrics("NCF", y_true, y_pred_ncf_binary)
print_metrics("LightFM", y_true, y_pred_lightfm_binary)

# Compute and print RMSE and MSE
rmse_ncf = np.sqrt(mean_squared_error(y_true, y_pred_ncf))
mse_ncf = mean_squared_error(y_true, y_pred_ncf)
rmse_lightfm = np.sqrt(mean_squared_error(y_true, y_pred_lightfm))
mse_lightfm = mean_squared_error(y_true, y_pred_lightfm)

print("\nRoot Mean Squared Error (RMSE) and Mean Squared Error (MSE):")
print(f"NCF - RMSE: {rmse_ncf:.4f}, MSE: {mse_ncf:.4f}")
print(f"LightFM - RMSE: {rmse_lightfm:.4f}, MSE: {mse_lightfm:.4f}")

import matplotlib.pyplot as plt

# Data
models = ['NCF', 'LightFM']
rmse_values = [rmse_ncf, rmse_lightfm]
mse_values = [mse_ncf, mse_lightfm]

# Plot RMSE
plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color=['blue', 'red'])
plt.xlabel('Models', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.title('Root Mean Squared Error (RMSE) Comparison', fontsize=16)
plt.show()

# Plot MSE
plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color=['blue', 'red'])
plt.xlabel('Models', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.title('Mean Squared Error (MSE) Comparison', fontsize=16)
plt.show()



# Compute Precision-Recall curve
precision_ncf, recall_ncf, _ = precision_recall_curve(data["test"].ravel(), data["ncf_predictions"].ravel())
precision_lightfm, recall_lightfm, _ = precision_recall_curve(data["test"].ravel(), data["lightfm_predictions"].ravel())

# Plot Precision-Recall curve
precision_recall(recall_ncf, precision_ncf,recall_lightfm, precision_lightfm)

# Compute ROC curve
fpr_ncf, tpr_ncf, _ = roc_curve(data["test"].ravel(), data["ncf_predictions"].ravel())
fpr_lightfm, tpr_lightfm, _ = roc_curve(data["test"].ravel(), data["lightfm_predictions"].ravel())

# Plot ROC curve
ROC_curve(fpr_ncf,tpr_ncf,fpr_lightfm,tpr_lightfm)



