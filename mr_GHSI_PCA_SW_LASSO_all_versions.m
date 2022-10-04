%% m/r GHSI + world data analysis (85 countries)
% Six approaches in the analysis of m/r dataset
% variables were already transformed and the outliers were removed
% (see mr_GHSI_dataset_transformations.m)
% Used functions: 
% relaxedlasso 
% relaxedlasso_max_sparse

% Approach 1 and 6 were used in all subsequent multivariate analyses

%% 1) detect + respond + health + risk + 18 demo variables (no PCA)

clear
rng default
load mr_GHSI_demo_data_transf_out.mat

mr = mr_GHSI_demo_data_transf_out(:,1); 
data_mat = mr_GHSI_demo_data_transf_out(:,[4,5,6,8,10:end]); 
data_varnames = mr_GHSI_demo_varnames([4,5,6,8,10:end]); 
mr_varname = mr_GHSI_demo_varnames(1) ; 

% Stepwise linear regression model:
swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

save V1_swlm swlm

% Lasso regression
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 

save V1_RelaxedLasso_Data B_fin FitInfo_fin LASSO_Results_table MSE_test_min sterr_MSE_test_min lambda_min

%% 2) Overall GHSI + 18 demo variables: 

clear
rng default
load mr_GHSI_demo_data_transf_out.mat

mr = mr_GHSI_demo_data_transf_out(:,1) ; 
data_mat = mr_GHSI_demo_data_transf_out(:,[2,10:end]) ; 
data_varnames = mr_GHSI_demo_varnames([2,10:end]) ; 
mr_varname = mr_GHSI_demo_varnames(1) ; 

% Stepwise linear regression model:
swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

save V2_swlm swlm

% Lasso regression
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 

save V2_RelaxedLasso_Data B_fin FitInfo_fin LASSO_Results_table MSE_test_min sterr_MSE_test_min lambda_min

%% 3) Covid + 18 demo variables

clear

load mr_GHSI_demo_data_transf_out.mat

mr = mr_GHSI_demo_data_transf_out(:,1) ; 
data_mat = mr_GHSI_demo_data_transf_out(:,[9:end]) ; 
data_varnames = mr_GHSI_demo_varnames([9:end]) ;
mr_varname = mr_GHSI_demo_varnames(1) ; 

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

save V3_swlm swlm
% 
% [B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 
% 
% save V3_RelaxedLasso_Data B_fin FitInfo_fin LASSO_Results_table MSE_test_min sterr_MSE_test_min lambda_min

% Sparse Relaxed Lasso regression: 

[B_fin , FitInfo_fin, MSE_lambda_mat,new_x, new_varnames ] = relaxedlasso_max_sparse(data_mat,mr, data_varnames)

save Sparse_Relaxed_Lasso_mr_V3 B_fin FitInfo_fin mr data_mat data_varnames MSE_lambda_mat new_x new_varnames
%% 4) detect + respond + health + PCA(risk + demo) up to 85% of variance explained 

clear
rng default
load mr_GHSI_demo_data_transf_out.mat

mr = mr_GHSI_demo_data_transf_out(:,1) ; 
mr_varname = mr_GHSI_demo_varnames(1) ;
index = mr_GHSI_demo_data_transf_out(:,[4,5,6]) ; 
index_varnames = mr_GHSI_demo_varnames([4,5,6]) ; 
index_norm = normalize(index) ; 

% Index PCA
[~,score,~,~,explained_ind,~] = pca(index_norm) ; 
disp(cumsum(explained_ind))
index_PC1 = score(:,1) ; 
index_PC2 = score(:,2) ; 

demo_data = mr_GHSI_demo_data_transf_out(:,[8,10:end]) ; 
demo_varnames = mr_GHSI_demo_varnames([8,10:end]) ; 
demo_data_norm = normalize(demo_data) ; 

% Demo data PCA: 
[coeff,score,~,~,explained_demo] = pca(demo_data_norm) ; 
disp(cumsum(explained_demo))
demo_pc = score(:,1:7) ; 
[R_all,P_all] = corrcoef([demo_pc,demo_data_norm]) ; 
R = R_all(8:end,1:7) ; 
P = P_all(8:end,1:7) ; 

%  %Correlation matrix: 
% corr_coef_table_v4 = array2table(R,'VariableNames',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7'},...
%     'RowNames',demo_varnames) 
% corr_pVal_table_v4 = array2table(P,'VariableNames',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7'},...
%     'RowNames',demo_varnames) 

% Matrix for the multivariate analysis: 
data_mat = [index_PC1, index_PC2 demo_pc] ; 
data_varnames = {'Index PC1','Index PC2' 'PC1','PC2','PC3','PC4','PC5','PC6','PC7'} ; 

swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

save V4a_swlm swlm

[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 

save V4a_RelaxedLasso_Data B_fin FitInfo_fin LASSO_Results_table MSE_test_min sterr_MSE_test_min lambda_min

%% 5) PCA(detect + respond + health + risk + 18 demo variables)

clear

load mr_GHSI_demo_data_transf_out.mat

mr = mr_GHSI_demo_data_transf_out(:,1) ; 
data_mat = mr_GHSI_demo_data_transf_out(:,[4,5,6,8,10:end]) ; 
data_varnames = mr_GHSI_demo_varnames([4,5,6,8,10:end]) ; 
mr_varname = mr_GHSI_demo_varnames(1) ; 

data_mat_norm = normalize(data_mat) ; 

% PCA

[coef,score, ~,~,explained] = pca(data_mat_norm) ; 
disp(sum(explained(1:8)))
demo_pc = score(:,1:8) ; 

% Correlation of relevant PCs with demo data: 

[R_all,P_all] = corrcoef([demo_pc,data_mat_norm]) ; 
R = R_all(9:end,1:8) ; 
P = P_all(9:end,1:8) ; 

% Correlation table: 
corr_coef_table_v5 = array2table(R,'VariableNames',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'},...
    'RowNames',data_varnames) 

corr_pVal_table_v5 = array2table(P,'VariableNames',{'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'},...
    'RowNames',data_varnames)
data_mat = demo_pc ; 
data_varnames = {'PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8'} ; 

% swlm = stepwiselm(data_mat,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])
% 
% save V5_swlm swlm
% 
% [B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 
% 
% save V5_RelaxedLasso_Data B_fin FitInfo_fin LASSO_Results_table MSE_test_min sterr_MSE_test_min lambda_min

[B_fin , FitInfo_fin, MSE_lambda_mat,new_x, new_varnames ] = relaxedlasso_max_sparse(data_mat,mr, data_varnames)

save Sparse_Relaxed_Lasso_mr_V5 B_fin FitInfo_fin mr data_mat data_varnames MSE_lambda_mat new_x new_varnames

%% 6) PC components on smaller number of variables (grouped variables)

clear
rng default
load mr_GHSI_demo_data_transf_out.mat

mr = mr_GHSI_demo_data_transf_out(:,1) ; 
mr_varname = mr_GHSI_demo_varnames(1) ; 

prosperity = mr_GHSI_demo_data_transf_out(:,[8,13,14,15]) ; 
prosperity_Varnames = {'Risk','IM','GDP','HDI'} ; 
prosperity_norm = normalize(prosperity) ; 
[~,score,~,~,explained_HDI,~] = pca(prosperity_norm) ; 
HDI_PC1 = score(:,1) ; 
cumsum(explained_HDI)
age = mr_GHSI_demo_data_transf_out(:,[12,18,19]) ; 
age_Varnames = {'MA','CH','AL'} ; 
age_norm = normalize(age) ; 
[~,score,~,~,explained_age,~] = pca(age_norm) ; 
age_PC1 = score(:,1) ; 
age_PC2 = score(:,2) ; 
cumsum(explained_age)
chronic = mr_GHSI_demo_data_transf_out(:,[22,23,27]) ;
chronic_varnames = {'CD','RBP','PL'} ; 
chronic_norm = normalize(chronic) ; 
[~,score,~,~,explained_ch,~] = pca(chronic_norm) ; 
cumsum(explained_ch)
chronic_PC1 = score(:,1) ; 
chronic_PC2 = score(:,2) ; 
other_variables = mr_GHSI_demo_data_transf_out(:,[4,5,6,10,11,16,17,20,21,24,25,26]) ; 
other_varnames = mr_GHSI_demo_varnames([4,5,6,10,11,16,17,20,21,24,25,26]) ; 
data_mat = [HDI_PC1, age_PC1, age_PC2,chronic_PC1, chronic_PC2, other_variables] ; 
data_mat_norm = normalize(data_mat) ; 
data_varnames = [{'HDI PC1', 'age PC1', 'age PC2', 'chronic PC1', 'chronic PC2'},other_varnames] ; 

% Stepwise linear regression: 
swlm = stepwiselm(data_mat_norm,mr,'Upper','linear','VarNames',[data_varnames,mr_varname])

save V6_swlm swlm

% Lasso regression: 
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(data_mat,mr, data_varnames) ; 

save V6_RelaxedLasso_Data B_fin FitInfo_fin LASSO_Results_table MSE_test_min sterr_MSE_test_min lambda_min







