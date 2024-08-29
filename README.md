# Feature Selection Methods Evaluation

This repository presents the evaluation of different feature selection methods applied to predictive models, including Logistic Regression (LR), Ridge Regression, and K-Nearest Neighbors (KNN). The feature selection methods evaluated include Sequential Feature Selection (SFS), Select K Best, and Recursive Feature Elimination (RFE). The results obtained from these methods are compared to a baseline model trained on the full feature set.

## Baseline Model Performance

The baseline model achieved an accuracy of **53.3301%**, which is below expectations. Notably, a simple predictor based solely on home team information predicts outcomes with an accuracy of **57.1686%**. This low performance of the baseline model underscores the need for feature selection to enhance the model's generalization ability and avoid overfitting.

## Feature Selection Methods and Results

### Logistic Regression (LR)

#### LR with Recursive Feature Elimination (RFE)
- **Accuracy:** 63.1162%
- **Summary:** This method provided a significant improvement over the baseline. The RFE method effectively reduced the feature set, improving model performance.

#### LR with Sequential Feature Selection (SFS)
- **Accuracy:** 62.0077%
- **Summary:** The SFS method improved the model's performance, though it was slightly less effective than the RFE method for Logistic Regression.

#### LR with Select K Best
- **Accuracy:** 63.4316%
- **Summary:** This method yielded the best accuracy among the Logistic Regression models. Select K Best focused on features with the highest ANOVA F-statistic values, leading to the most accurate predictions.

### Ridge Regression

#### Ridge Regression with Sequential Feature Selection (SFS)
- **Accuracy:** 63.1612%
- **Summary:** The SFS method significantly improved the model's performance compared to the baseline. The algorithm selected 30 features that most effectively contributed to the improvement of prediction accuracy.

#### Ridge Regression with Select K Best
- **Accuracy:** 63.4045% (Highest among tested methods for Ridge Regression)
- **Summary:** The model achieved the highest accuracy using this method. Select K Best focused on features with the highest ANOVA F-statistic values, indicating a strong relationship with the target classes.

#### Ridge Regression with Recursive Feature Elimination (RFE)
- **Accuracy:** 62.9179%
- **Summary:** RFE also improved the model's accuracy compared to the baseline, though the result was slightly lower than the other two methods.

### K-Nearest Neighbors (KNN)

#### KNN with Select K Best
- **Accuracy:** 62.7647%
- **Summary:** The Select K Best method also worked well with the KNN model, resulting in a solid improvement over the baseline.

#### KNN with Sequential Feature Selection (SFS)
- **Accuracy:** 62.5211%
- **Summary:** While SFS provided a performance boost, it was less efficient in this context compared to other methods.

#### KNN with Recursive Feature Elimination (RFE)
- **Accuracy:** 56.5648%
- **Summary:** This method provided the least improvement among the evaluated methods for KNN, though it still surpassed the baseline.

## Conclusion

All feature selection methods significantly enhanced the model's accuracy compared to the baseline. For Logistic Regression, the **Select K Best** algorithm yielded the highest accuracy, making it the most effective in identifying critical features. The Ridge Regression models also showed strong performance, with the **Select K Best** method achieving the highest accuracy among Ridge Regression models. The KNN models benefited from feature selection, though the improvements were less consistent across methods. Overall, feature selection proves to be a valuable step in improving model accuracy and performance.
