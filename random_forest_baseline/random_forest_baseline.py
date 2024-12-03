import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score

RANDOM_STATE = 0
TEST_SIZE = 0.1
VAL_SIZE = 0.1

def RF(X, y, X_test):
    model = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                   max_depth=10, min_samples_leaf=1,
                                   max_features=0.4, n_jobs=3)
    model.fit(X, y)
    preds = model.predict_proba(X_test)[:, 1]

    return preds

data = pd.read_csv("train.csv")  # Replace with the dataset path

# Keep only necessary columns
data = data[['question1', 'question2', 'is_duplicate']]

df_train = pd.concat([
    data, pd.read_csv("features12.csv")   
], axis=1)

X = df_train.drop(['question1', 'question2', 'is_duplicate'], axis=1).values
y = df_train['is_duplicate'].values

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

X_training, X_val, y_training, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)

RF_preds_val = RF(X_training, y_training, X_val)

val_log_loss_score = log_loss(y_val, RF_preds_val)
print('Validation log loss = {}'.format(val_log_loss_score))
RF_preds_val[RF_preds_val > 0.5] = 1
RF_preds_val[RF_preds_val <= 0.5] = 0
accuracy = accuracy_score(y_val, RF_preds_val)
print('Validation accuracy = {}\n'.format(accuracy))

print('Predicting...')
RF_preds_test = RF(X_training, y_training, X_test)

# evaluate testing set using log loss and accuracy metrics
test_log_loss_score = log_loss(y_test, RF_preds_test)
print('Testing log loss = {}'.format(test_log_loss_score))
RF_preds_test[RF_preds_test > 0.5] = 1
RF_preds_test[RF_preds_test <= 0.5] = 0
accuracy = accuracy_score(y_test, RF_preds_test)
print('Testing accuracy = {}\n'.format(accuracy))

"""
Training data shape after rebalancing:(310027, 15)
Validation log loss = 0.4595494660105385
Validation accuracy = 0.8222771744973659

Predicting...
Testing log loss = 0.4593044236938024
Testing accuracy = 0.8235009515208206
"""

"""
Validation log loss = 0.6509724027474957
Validation accuracy = 0.6292082337098415

Predicting...

Testing log loss = 0.6478249665474349
Testing accuracy = 0.6314526701130376
"""