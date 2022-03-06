## Predicting bank users behaviour

#### Introduction and EDA

The provided dataset consists of 15 columns: 14 features and 1 target.

The features are mostly categorical (10) and contained specific and limited information about phone calls used to assess if a product would be subscribed.

The target is imbalanced, indicating that most of the clients did not subscribe. In particular only around 11% of the records had a positive target, out of the whole training set.

#### Data preparation

Numerical features are left as-is, because scaling them did not improve the score metrics of the model. On the other side, target encoding is performed on categorical features. In this case it was important to execute target encoding after train-test split to avoid possible data leakage, so the mean of the target of the training set is used to encode also validation and test set.

Correlation matrix after target encoding shows that "duration" is the feature which is most correlated with the target, while  there is strong correlation between "previous" and "te_poutcome". Removing some correlated features did not improve the score of the model.

In order to deal with class imbalance, target dataset is rebalanced with a factor of 0.15% and different experiments were carried out.

#### Models implementation and evaluation

Three models were compared and evaluated without parameter tuning:

- Random Forest:
    - roc_auc_score (without rebalance): 0.9250
    - roc_auc_score (with rebalance): 0.9263
    
- Logistic Regression:
    - roc_auc_score (without rebalance): 0.9165
    - roc_auc_score (with rebalance): 0.9165

- XGBoost:
    - roc_auc_score (without rebalance): 0.9250
    - roc_auc_score (with rebalance): 0.9226

Random forest and XGBoost appear to have better scores compared to Logistic regression so they were both optimezed. Random forest increased performance on rebalanced dataset.

Evaluation after optimizazion:

- Random Forest (on rebalanced training set)
    - roc_auc_score: 0.9306
    - pr_auc_score: 0.6168
    
- XGBoost (on original training set)
    - roc_auc_score: 0.9288
    - pr_auc_score: 0.5426

After parameters tuning, the model with highest performance appears to be the Random Forest despite also XGBoost has high predictive capacity on the target. Note: pr_auc_score is not computed as area under pr curve but with an approximation using average_precision_score.

Using tuned xgboost model, lift score was implemented. it is interesting to notice that this score indicates that if the 10th decile (the one with highest probbility to subscribe) is taken into account to perform a certain marketing action, we would have more than 5 times chances to perform the action on a subscriber compared to taking random clients.

#### Deep learning implementation

A test was performed also using Neural networks: the base model showed poorer performance compared to the other models; the performance increased after parameter tuning but did not overperform Random Forest or XGBoost.

- before parameters tuning - roc_auc_score: 0.9036

- after parameters tuning - roc_auc_score: 0.9172

#### Feature importance

The independent features which were most impactful on the target are duration, te_poutcome, te_month, te_contact, age.
In particulare from the shap values analysis it is possible to say that high values of duration or age correspond to a positive target. 
Also, higher value of te_contact and te_month, so "cellular" or the months around "december", rise the model's propensity to predict a positive target.

#### Final outcome

Random forest was finally used to predict the target values on the test dataset. Out of 40 records, 4 appear to have the highest probability to be a subscriber, using 0,48 as a probability threshold.

#### Possible next steps

A better parameter tuning could be done on neural network as there was a good improvement on the metric, despiete not being enough to overperform the other models.
New features could be implemented and a feature selection using PCA could also be tried to see if performances could be improved.