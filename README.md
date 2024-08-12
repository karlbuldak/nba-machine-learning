# Feature Selection Methods Evaluation

This repo presents the evaluation of different feature selection methods applied to a predictive model. The methods evaluated include Sequential Feature Selection (SFS), Select K Best, and Recursive Feature Elimination (RFE). The results obtained from these methods are compared to a baseline model trained on the full feature set.

## Baseline Model Performance

The baseline model achieved an accuracy of **53.3301%**, which is below expectations. Notably, a simple predictor based solely on home team information predicts outcomes with an accuracy of **57.1686%**. This low performance of the baseline model underscores the need for feature selection to enhance the model's generalization ability and avoid overfitting.

## Feature Selection Methods and Results

### Sequential Feature Selection (SFS)
- **Accuracy:** 63.1612%
- **Number of Features Selected:** 30
- **Summary:** This method significantly improved the model's performance compared to the baseline. The algorithm selected 30 features that most effectively contributed to the improvement of prediction accuracy.

### Select K Best
- **Accuracy:** 63.4045% (Highest among tested methods)
- **Number of Features Selected:** 30
- **Summary:** The model achieved the highest accuracy using this method. Select K Best focused on features with the highest ANOVA F-statistic values, indicating a strong relationship with the target classes.

### Recursive Feature Elimination (RFE)
- **Accuracy:** 62.9179%
- **Number of Features Selected:** Varies depending on the configuration
- **Summary:** RFE also improved the model's accuracy compared to the baseline, though the result was slightly lower than the other two methods.

## Conclusion

All three feature selection methods significantly enhanced the model's accuracy compared to the baseline. The Select K Best algorithm yielded the best results, suggesting that this method is the most effective in identifying the key features critical for predicting match outcomes. The results obtained from Sequential Feature Selection and Recursive Feature Elimination are comparable and also substantially exceed the baseline model's accuracy.
