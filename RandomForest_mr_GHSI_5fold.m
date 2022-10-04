%% Random Forest regression
% m/r GHSI dataset (V1 and V6)

clear

% For reproducibility: 
rng default

% Load data
load mr_GHSI_demo_data_transf_out.mat
mr = mr_GHSI_demo_data_transf_out(:,1); 
data_mat = mr_GHSI_demo_data_transf_out(:,[4,5,6,8,10:end]); 
data_varnames = mr_GHSI_demo_varnames([4,5,6,8,10:end]); 
mr_varname = mr_GHSI_demo_varnames(1); 

Varnames = data_varnames;  
matrix_all = data_mat; 

% CV parameters:
n_fold = 5;
n_repart = 40;
cv = cvpartition( mr,'KFold',n_fold,'Stratify',false);

% Random Forest hyperparameters: 
maxNumSplits = [3 6 9 12 18 20 22 24 26 30 35 37 40 43 45 47 50];
minLeafSize = 1:20;
NTrees_vec =  [5 8 10 14 18 26 50 106 133 160 193 226 263 300 350 400 450];

% Results matrix: 
MSE_alpha_lambda_mat = [];

for NumSplits = maxNumSplits
    for LeafSize = minLeafSize
        for NTrees = NTrees_vec
            SE_vector_train = [];
            SE_vector_test = [];
            % Cross-validation:
            for repart = 1:n_repart 
                cnew = repartition( cv );
                parfor  i=1:n_fold
                        % Test and train set: 
                        index_train = find(cnew.test(i)==0);
                        index_test = find(cnew.test(i));
                        mr_train = mr(index_train);
                        matrix_all_train = matrix_all(index_train,:);
                        % Choose input variables with (any) correlation
                        % P-value < 0.1
                        [~,P_Pearson] = corrcoef([mr_train matrix_all_train]);
                        [~,P_Kendall] = corr([mr_train matrix_all_train],'Type','Kendall');
                        [~,P_Sperman] = corr([mr_train matrix_all_train],'Type','Spearman');
                        index_Pearson = find(P_Pearson(2:end,1) < 0.1);
                        index_Kendall = find(P_Kendall(2:end,1) < 0.1);
                        index_Spearman = find(P_Sperman(2:end,1) < 0.1);
                        index_all_reduced = unique([index_Pearson' index_Kendall' index_Spearman']);
                        % train matrix with reduced number of input variables:
                        matrix_all_train_reduced = matrix_all_train(:,index_all_reduced);
                        % train set standardization: 
                        mean_matrix_all_train = mean(matrix_all_train_reduced);
                        std_matrix_all_train = std(matrix_all_train_reduced);
                        matrix_all_train_norm = (matrix_all_train_reduced - mean_matrix_all_train )./std_matrix_all_train;
                        % test set: 
                        mr_test = mr(index_test);
                        matrix_all_test = matrix_all(index_test,:);
                        matrix_all_test_reduced = matrix_all_test(:,index_all_reduced);
                        matrix_all_test_norm = (matrix_all_test_reduced - mean_matrix_all_train)./std_matrix_all_train;
                        % Random Forest regression 
                        t = templateTree('MaxNumSplits',NumSplits,'MinLeafSize',LeafSize);
                        MdlOpt_train = fitrensemble(matrix_all_train_norm,mr_train,'Method','bag','NumLearningCycles',NTrees,'Learners',t);   
                        mrhat_train = predict(MdlOpt_train,matrix_all_train_norm);
                        % MSE on train set:
                        MSE_train_norm = sum((mr_train - mrhat_train).^2)/sum((mr_train - mean(mr_train)).^2);
                        mrhat_test = predict(MdlOpt_train,matrix_all_test_norm);
                        % MSE on test set:
                        MSE_test_norm = sum((mr_test - mrhat_test).^2)/sum((mr_test - mean(mr_test)).^2);
                        SE_vector_test = [SE_vector_test MSE_test_norm];
                        SE_vector_train = [SE_vector_train MSE_train_norm]; 
                end
            end
            MSE_train = mean(SE_vector_train);
            MSE_test =  mean(SE_vector_test);
            sterr_MSE_train = 2*std(SE_vector_train )/sqrt(length(SE_vector_train));
            sterr_MSE_test = 2*std(SE_vector_test )/sqrt(length(SE_vector_test));
            MSE_alpha_lambda_mat = [MSE_alpha_lambda_mat [NumSplits;LeafSize;NTrees;MSE_train;sterr_MSE_train;MSE_test;sterr_MSE_test]];
        end
    end
end
  
% Find min MSE model: 
[MSE_test_min,index_min] = min(MSE_alpha_lambda_mat(6,:));
% min MSE model parameters: 
NumSplits_min = MSE_alpha_lambda_mat(1,index_min);
LeafSize_min = MSE_alpha_lambda_mat(2,index_min);
Ntrees_min = MSE_alpha_lambda_mat(3,index_min);
MSE_train_min = MSE_alpha_lambda_mat(4,index_min);
sterr_MSE_train_min = MSE_alpha_lambda_mat(5,index_min );   
MSE_test_min = MSE_alpha_lambda_mat(6,index_min);
sterr_MSE_test_min = MSE_alpha_lambda_mat(7,index_min);

% Select variables for final RF model: 
[R_Pearson,P_Pearson] = corrcoef([mr matrix_all]);
[R_Kendall,P_Kendall] = corr([mr matrix_all],'Type','Kendall');
[R_Sperman,P_Sperman] = corr([mr matrix_all],'Type','Spearman');
index_Pearson = find(P_Pearson(2:end,1) < 0.1);
index_Kendall = find(P_Kendall(2:end,1) < 0.1);
index_Spearman = find(P_Sperman(2:end,1) < 0.1);
index_all_reduced = unique([index_Pearson' index_Kendall' index_Spearman']);
Varnames_reduced = Varnames(index_all_reduced);                   
matrix_all_reduced = matrix_all(:,index_all_reduced);
% Standardization: 
mean_matrix_all = mean(matrix_all_reduced);
std_matrix_all = std(matrix_all_reduced);
matrix_all_norm = (matrix_all_reduced - mean_matrix_all)./std_matrix_all;
% Final RF regression (on entire dataset):
t_all = templateTree('MaxNumSplits',NumSplits_min,'MinLeafSize',LeafSize_min);
MdlOpt_all = fitrensemble(matrix_all_norm,mr,'Method','bag','NumLearningCycles', Ntrees_min,'Learners',t_all);
mrhat = predict(MdlOpt_all,matrix_all_norm);
R_square = 1 - (sum((mr - mrhat ).^2)/sum((mr - mean(mr)).^2));
imp = predictorImportance(MdlOpt_all); 
% Display results
fprintf('<strong> Random Forest V1 results:</strong>\n\n')
fprintf('min MSE model parameters: \nN_trees = %d \nNSplits = %d\nLeafSize = %d \n',...
    Ntrees_min,NumSplits_min,LeafSize_min) ; 
fprintf('minimal MSE (train set)= %d , with SE = %d \n',MSE_train_min,sterr_MSE_train_min)
fprintf('minimal MSE (test set)= %d , with SE = %d \n',MSE_test_min,sterr_MSE_test_min)
fprintf('R_square = %.4f\n', R_square) ; 
figure;
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
h.XTickLabel = Varnames_reduced  ;
h.XTickLabelRotation = 45 ;
h.TickLabelInterpreter =  'none' ;

save RF_5fold_repart_results_V1_mr_world  MdlOpt_all  imp  MSE_alpha_lambda_mat  matrix_all  mr  Varnames_reduced Varnames

%% Hyperparameter optimization:

upperlimit = MSE_test_min + sterr_MSE_test_min ; 
opt_ind = find(MSE_alpha_lambda_mat( 6 , : ) < upperlimit);

subplot(1,3,1)
histogram(MSE_alpha_lambda_mat( 1 , opt_ind ))
xlabel('Maximal number of splits')
subplot(1,3,2)
histogram(MSE_alpha_lambda_mat( 2 , opt_ind ))
xlabel('Minimal nimber of leaves')
subplot(1,3,3)
histogram(MSE_alpha_lambda_mat( 3 , opt_ind ))
xlabel('Numbar of trees')




