%% Excess and Unexplained deaths data transformation

% Excess and unexplained deaths + Covid numbers + 18 demo parameters
% for 59 countries

% Load data
clear
load ExcessDeathsData.mat 
load excess_deaths_additional_data
load excess_deaths_additional_data_2

covid_deaths = excess_deaths_additional_data(:,3); 
covid_cases = excess_deaths_additional_data(:,2); 
CFR = covid_deaths./covid_cases;  
total_deaths = excess_deaths_additional_data_2(:,1); % not only COVID
excess_deaths = excess_deaths_additional_data_2(:,2); % compared to baseline 
population_size = excess_deaths_additional_data_2(:,3); 
days = excess_deaths_additional_data_2(:,4); 
excess_deaths_additional_varnames = {'R0', 'Total cases','Total deaths', 'max daily cases', 'max daily deaths'} ; 

% Calculate excess and unexplained deaths:
unexplained_deaths = (excess_deaths - covid_deaths)./(total_deaths - excess_deaths) ; 
relative_excess_deaths = excess_deaths./(total_deaths - excess_deaths) ; 

% Data transformations: 

mat = ExcessDeaths_full_data_matrix ; 
mat_transf = zeros(size(mat)) ; 

mat_transf(:,1) = log(mat(:,1) - min(mat(:,1))) ; % Excess deaths 1 
mat_transf(:,2) = log(mat(:,2) - min(mat(:,2))) ; % Excess deaths 2 
mat_transf(:,3) = log(mat(:,3)) ; % m/r
mat_transf(:,4) = (mat(:,4)).^(1/3) ; % Overall
mat_transf(:,5) = mat(:,5) ; % Prevent
mat_transf(:,6) = mat(:,6) ; % Detect
mat_transf(:,7) = (mat(:,7)).^(1/3) ; % Respond
mat_transf(:,8) = mat(:,8) ; % Health
mat_transf(:,9) = mat(:,9) ; % Norms
mat_transf(:,10) = mat(:,10) ; % Risk
mat_transf(:,11) = sqrt(mat(:,11)) ; % Covid
mat_transf(:,12) = log(mat(:,12)) ; % BUAPC
mat_transf(:,13) = (mat(:,13)).^2 ; % UP
mat_transf(:,14) = - log(max(mat(:,14))- mat(:,14)) ; % MA
mat_transf(:,15) = log(mat(:,15)) ; % IM
mat_transf(:,16) = log(mat(:,16)) ; % GDP
mat_transf(:,17) = (mat(:,17)).^2 ; %HDI
mat_transf(:,18) = mat(:,18) ; % IE
mat_transf(:,19) = log(mat(:,19)) ; % RE 
mat_transf(:,20) = (mat(:,20)).^2 ; %CH
mat_transf(:,21) = (mat(:,21)).^2 ; %AL
mat_transf(:,22) = (mat(:,22)).^(1/3) ; % OB
mat_transf(:,23) = mat(:,23) ; % SM
mat_transf(:,24) = log(mat(:,24)) ; % CD
mat_transf(:,25) = log(mat(:,25)) ; % RBP
mat_transf(:,26) = mat(:,26) ; % IN
mat_transf(:,27) = - sqrt(max(mat(:,27))-mat(:,27)) ; % BCG
mat_transf(:,28) = mat(:,28) ; % ON
mat_transf(:,29) = log(mat(:,29)) ; % PL

% Additional data: 
mat_add = [relative_excess_deaths unexplained_deaths CFR excess_deaths_additional_data  ] ; 
names_add = {'excess deaths', 'unexplained deaths', 'CFR','R0', 'Average daily cases','Average daily deaths', 'Max daily cases', 'Max daily deaths'} ; 

% Calculate relative values of additional variables:  
mat_add(:,5) = mat_add(:,5)./(population_size.*days) ; % Total cases
mat_add(:,6) = mat_add(:,6)./(population_size.*days) ; % Total deaths
mat_add(:,7) = mat_add(:,7)./population_size ; % max daily cases
mat_add(:,8) = mat_add(:,8)./population_size ; % max daily deaths
max_deaths_sort = sort(mat_add(:,8)) ; % sort data
mat_add(42,8) = max_deaths_sort(2) ; % Substitute value 0 (for Monacco) with second smallest value

% Additional data transformation:
mat_add_transf(:,1) = sqrt(mat_add(:,1)-min(mat_add(:,1))) ; % relative excess deaths
mat_add_transf(:,2) = (mat_add(:,2)-min(mat_add(:,2))).^(1/3) ; % unexplained deaths
mat_add_transf(:,3) = log(mat_add(:,3)) ; % CFR
mat_add_transf(:,4) = log(mat_add(:,4)) ; % R0
mat_add_transf(:,5) = log(mat_add(:,5)) ; % Average daily cases (Total cases po danu po stanovniku)
mat_add_transf(:,6) = log(mat_add(:,6)) ; % Average daily deaths (Total deaths po danu po stanovniku)
mat_add_transf(:,7) = log(mat_add(:,7)) ; % Max daily cases po stanovniku
mat_add_transf(:,8) = log(mat_add(:,8)) ; % Max daily deaths po stanovniku

mat_transf_fin = [mat_add_transf, mat_transf(:,4:end)] ; 
Varnames = [names_add ExcessDeaths_full_Varnames(4:end) ] ; 
mat_transf_out = substituteoutlier(mat_transf_fin) ; 

save excess_deaths_transf_out_data_NEW mat_transf_out Varnames

%% Univariate analysis

clear

load excess_deaths_transf_out_data_NEW.mat

excess_deaths = mat_transf_out(:,1) ; 
unexplained_deaths = mat_transf_out(:,2) ; 
CFR = mat_transf_out(:,3) ; 
R0 = mat_transf_out(:,4) ;

%% COVID Counts PCA:
counts = mat_transf_out(:, 5:8) ; 
counts_norm = normalize(counts) ; 
counts_varnames = Varnames(5:8) ; 
[coeff_counts, score_counts, ~,~,explained_counts] = pca(counts_norm) ; 
s = cumsum(explained_counts) ;
disp(s(1))
counts_PC1 = score_counts(:,1) ; % 88.9871% variablity 
figure
pc_correlation_plots(score_counts,counts_norm,1,'counts',counts_varnames,1,1)


%% GHSI PCA:
GHSI = mat_transf_out(:,[9:14 16]) ;  % Svi GHSI but Risk, which is included in demo variables
GHSI_norm = normalize(GHSI) ; 
GHSI_varnames = Varnames([9:14 16]) ; 
[coeff_GHSI, score_GHSI, ~,~,explained_GHSI] = pca(GHSI_norm) ; 
disp(cumsum(explained_GHSI)) ;

GHSI_PC1 = score_GHSI(:,1) ; % 79% variability
GHSI_PC2 = score_GHSI(:,2) ; % 85.6% variability
figure
pc_correlation_plots(score_GHSI,GHSI_norm,2,'GHSI',GHSI_varnames,1,2)

%% demo PCA
demo = mat_transf_out(:,[15 17:end]) ; 
demo_norm = normalize(demo) ; 
demo_varnames = Varnames([15 17:end]) ; 
[coeff_demo, score_demo, ~,~,explained_demo] = pca(demo_norm) ; 
s = cumsum(explained_demo) ;
disp(s(8))
demo_PCs = score_demo(:,1:8) ; 
figure
pc_correlation_plots(score_demo,demo_norm,8,'demo',demo_varnames,4,2)
%% Predictor matrix
predictor_mat = [CFR R0 counts_PC1 GHSI_PC1 GHSI_PC2 demo_PCs]  ; 
predictor_varnames = {'CFR', 'R0', 'counts PC1', 'GHSI PC1', 'GHSI PC2',...
    'demo PC1','demo PC2','demo PC3','demo PC4','demo PC5','demo PC6',...
    'demo PC7','demo PC8'} ; 
response_varnames = {'excess deaths', 'unexplained deaths'} ; 

save excess_deaths_PC_predictors predictor_mat predictor_varnames response_varnames excess_deaths unexplained_deaths

%% Excess deaths regressions
clear
load excess_deaths_PC_predictors

% linear regression model
lin_mdl = fitlm(normalize(predictor_mat),excess_deaths,'VarNames',[predictor_varnames,response_varnames(1)]) ; 

% stepwise linear regression
swlm = stepwiselm(normalize(predictor_mat),excess_deaths,'Upper','linear','VarNames',[predictor_varnames,response_varnames(1)]) ; 

% Lasso regression
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(predictor_mat,excess_deaths, predictor_varnames) ; 

%% Unexplained deaths regressions
clear
load excess_deaths_PC_predictors

% Linear regression model
lin_mdl = fitlm(normalize(predictor_mat),unexplained_deaths,'VarNames',[predictor_varnames,response_varnames(2)]) 

% Stepwise linear regression
swlm = stepwiselm(normalize(predictor_mat),unexplained_deaths,'Upper','linear','VarNames',[predictor_varnames,response_varnames(2)]) 
figure
cat = categorical(swlm.PredictorNames); 
cat = reordercats(cat, swlm.PredictorNames);
bar(cat,swlm.Coefficients.Estimate(2:end),'FaceColor', BFC,'EdgeColor',BEC,'LineWidth',1.5);
title('Stepwise linear regression');
ylabel('Model coefficients');

% Lasso regression
[B_fin , FitInfo_fin, LASSO_Results_table, MSE_test_min, sterr_MSE_test_min, lambda_min ]  = relaxedlasso(predictor_mat,unexplained_deaths, predictor_varnames)  ; 
