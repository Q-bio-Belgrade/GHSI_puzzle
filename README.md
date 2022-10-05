# GHSI_puzzle
Code for "GHSI COVID-19 puzzle: did highly developed countries indeed fare worse?"

MATLAB scripts:

GHSI_mr_dataset_transformations.m : Transformations for the m/r dataset (mr_GHSI_demo_data.mat) and removal of the outliers (by substituting them with the transformed variable median value); Pearson's correlation of all variables; Correlation of m/r and selected GHS Index categories.

mr_GHSI_PCA_SW_LASSO_all_versions.m : PCA, stepwise linear regression and lasso regression on 6 versions transformed m/r dataset (mr_GHSI_demo_data_transformed_out.mat). Note that versions 1 (4 GHSI + 18 demo variables) and 6 (Variables and PCs) were used in the article, while lasso and elastic net results from version 3 are presented in the Supplementary Materials (Supplementary Figure 3).

Excess_deaths_data_analysis.m : assembling of the excess / unexplained deaths dataset; data transformations; PCA; univariate regression (Pearson's correlation); linear regression model; stepwise linear regression; relaxed lasso regression. 

Elastic_Net_Sparse_mr_GHSI.m : Elastic net regression in which sparsest model (among the models with minimal MSE) is chosen. This code was used in all implementations of elastic net regressions (with versions 1 and 6 of m/r dataset, excess and unexplained deaths dataset). Here, the m/r dataset with variables and PCs (V6) is presented.

Elastic_Net_Relaxed_Sparse_mr_GHSI.m : Second, relaxed round of Elastic Net regression. Uses variables selected in the first round as input. Also selects sparsest model. This code was used in all implementations of elastic net regressions (with versions 1 and 6 of m/r dataset, excess and unexplained deaths dataset). Here, the m/r dataset with variables and PCs (V6) is presented.

RandomForest_mr_GHSI_5fold.m : Random Foerst regression with 5-fold cross validation. Modifications of this code were used for all RF regressions in the research.

GBoost_mr_GHSI_5fold.m :  Gradient Boost regression with 5-fold cross validation. Modifications of this code were used for all GB regressions in the research.

supervised_pca_mr_GHSI_demo.m : Supervised PCA followed by linear regression model; Performed only on 4 GHSI + 18 demo variables with m/r as the response variable. 

MATLAB functions:

substituteoutlier.m : substitute outliers with the variable median

relaxedlasso.m : relaxed lasso regression (min MSE model is selected as the final model in both rounds)

relaxedlasso_max_sparse.m : relaxed lasso regression (sparsest model is selected as the final model in both rounds)

pc_correlation_plots.m : calculates the correlation of given principal components with the variables entering PCA and returns panel of correlation plots


Preprint of the article is available on MedRxiv: https://www.medrxiv.org/content/10.1101/2022.08.28.22279258v1

