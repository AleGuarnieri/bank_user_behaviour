### Introduction

The provided dataset consists of 15 columns: 14 features and 1 target.

The features are mostly categorical (10) and contained specific information about people interviewed during phone calls. These phone calls were part of a marketing campaign used to assess if a product would be subscribed by a client.

Goal of the project is to implement and evaluate a model to predict future subscriptions of clients.

### Data preparation and EDA

The preoblem can be considered a supervised binary classification prediction problem. 

Data did not show issues or particular outliers to be removed.

Numerical features are left as-is, because scaling them did not improve the score metrics of the model. On the other side, target encoding is performed on categorical features. In this case it was important to execute target encoding after train-test split to avoid possible data leakage, so the mean of the target on each feature of the training set is used to encode also validation and test set, assuming that the distribution of the target in the training set will be similar into validation and test sets.

Correlation matrix after target encoding shows that "duration" is the feature which is most correlated with the target, while  there is strong correlation between "previous" and "te_poutcome". Removing some correlated features did not improve the score of the model.

The target is imbalanced, indicating that most of the clients did not subscribe. In particular only around 11% of the records had a positive target, out of the whole training set.
In order to deal with this problem, target dataset is rebalanced with a factor of 15%. Different experiments showed that rebalancing did not improve the score on all the models, only on Random Forest.

From the above graphs, the month with more subscription is May while the one with less subscriptions is December. Celullar contacts always bring more subscriptions across all months. The most frequent "outcome" is "nonexistent".

### Models implementation and evaluation

Three base models were compared and evaluated:

- Random Forest:
    - roc_auc_score (without rebalance): 0.9251
    - roc_auc_score (with rebalance): 0.9263
    
- Logistic Regression:
    - roc_auc_score (without rebalance): 0.9166
    - roc_auc_score (with rebalance): 0.9166

- XGBoost:
    - roc_auc_score (without rebalance): 0.9250
    - roc_auc_score (with rebalance): 0.9226

Random forest and XGBoost appear to have better scores compared to Logistic regression. Random forest increased performance on rebalanced dataset. 

Both Random Forest and XGBoost were optimezed using cross validation. Here the metrics after hyperparameter tuning:

- Random Forest (on rebalanced training set)
    - roc_auc_score: 0.9307
    - pr_auc_score: 0.6173
    
- XGBoost (on original training set)
    - roc_auc_score: 0.9288
    - pr_auc_score: 0.5985

The model with highest performance appears to be the Random Forest despite also XGBoost performs well. Note: pr_auc_score is not computed as area under pr curve but with an approximation using average_precision_score.

Using tuned xgboost model, lift score was calculated. it is interesting to notice that this score indicates that if the 10th decile (the one with highest probbility to subscribe) is taken into account to perform a certain marketing action, we would have more than 5 times chances to perform this action on a possible subscriber compared to taking random clients.

### Deep learning implementation

A test was performed also using Neural networks: the base model showed poorer performance compared to the other models; the performance increased after parameter tuning but did not overperform Random Forest or XGBoost.

- Before parameters tuning - roc_auc_score: 0.9037

- After parameters tuning - roc_auc_score: 0.9172

### Feature importance

The independent features which were most impactful on the target are duration, te_poutcome, te_month, te_contact, age.
In particulare from the shap values analysis it is possible to say that high values of duration or age correspond to predicting a positive target. 

Also, higher value of te_contact and te_month, so "cellular" or months like "october" or "december", rise the model's propensity to predict a positive target. 

On the other side, low values of "campaign", which are the most frequent ones among this feature values, have an impact on predicting a 0 target.

### Final outcome

Random forest was finally used to predict the target values on the test dataset. Out of 40 records, 4 appear to have the highest probability to be a subscriber, using 0,5 as a probability threshold.

Results were saved in these files:
- random_forest_predictions.csv: it contains only 0 or 1 for each test record, in the same order of the test records
- random_forest_test_with_predict_proba.csv: it contains test records and related predicted probability

### Possible next steps

A better parameter tuning could be done on neural network as there was a good improvement on the metric, despite not being enough to overperform the other models.
New features could be engineered and a feature selection algorithm like PCA could be tried to check if performance of some models could be improved even more.

The entire notebook could be organized into a python script so that if new data come, the process of data preparation, modeling and evaluation could be repeated and executed constantly.