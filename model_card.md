# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model was automatically generated using LightAutoMl.

Model is a blended model with the following components:
- 0.09145 * (5 averaged models Lvl_0_Pipe_1_Mod_0_LightGBM)
- 0.72006 * (5 averaged models Lvl_0_Pipe_1_Mod_1_Tuned_LightGBM)
- 0.18849 * (5 averaged models Lvl_0_Pipe_1_Mod_2_CatBoost)

This model was chosen based on 5-fold cross-validation (CV) on the training set.

## Intended Use

The model is only intended as an example to demonstrate deploying and serving of a machine learning model.
It is not intended for use in production.

## Training Data

80% of the dataset was used for training.

## Evaluation Data

20% of the dataset was used for evaluation.

## Metrics
|  Precision   |  Recall  |  F-beta  |
|:------------:|:--------:|:--------:|
 |   0.81106    | 0.672183 | 0.73512  |

## Ethical Considerations

The training dataset is based on the Census database and was collected in 1994. 
It is therefore most likely outdated and the resulting model should not be used for production purposes.

## Caveats and Recommendations

The model should be used with caution since it will reflect the bias of the dataset.