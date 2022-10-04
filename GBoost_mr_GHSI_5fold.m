%% GradientBoost m/r GHSI 5 fold corss-validation x 40 repart
% m/r GHSI dataset (V1 and V6)
% **V1 in this example**
clear
rng default
load mr_GHSI_demo_data_transf_out.mat

mr = mr_GHSI_demo_data_transf_out(:,1); 
data_mat = mr_GHSI_demo_data_transf_out(:,[4,5,6,8,10:end]); 
data_varnames = mr_GHSI_demo_varnames([4,5,6,8,10:end]); 
mr_varname = mr_GHSI_demo_varnames(1); 

Varnames = data_varnames ; 
matrix_all = data_mat ; 

% CV parameters: 
n_fold = 5;
n_repart = 40;
cv = cvpartition(mr,'KFold',n_fold,'Stratify',false);

% Gradient Boost hyperparameters: 
learnRate = [0.1 0.20 0.25 0.35 0.5 0.75 1] ;
maxNumSplits = [1 2 3 4 5 6 7 8 10 12 14 16 24 32];
minLeafSize = [1 2 3 4 5 8 12 16 18 20 22 25];
NTrees_vec = [5 10 13 16 21 27 38 65 92];

% Results matrix: 
MSE_alpha_lambda_mat = [];

% Parameter combinations matrix:
param_matrix = []   ;

for NumSplits = maxNumSplits
    for LeafSize = minLeafSize
        for Lrate = learnRate
            for NTrees = NTrees_vec
                param_matrix = [param_matrix [NumSplits;LeafSize;Lrate;NTrees]];        
            end
        end
    end
end

% Random permutation of param_matrix indices:
lena_param = size( param_matrix , 2 );
param_index_perm_vec = randperm(lena_param);

for index = param_index_perm_vec
    % Hyperparameter values:
    NumSplits = param_matrix(1,index);
    LeafSize = param_matrix(2,index);
    Lrate = param_matrix(3,index);
    NTrees = param_matrix(4,index);
    % SE vectors for given hyperparameter combination:
    SE_vector_train = [];
    SE_vector_test = [];
    for repart = 1:n_repart 
        cnew = repartition( cv );
        parfor  i=1:n_fold
                index_train = find(cnew.test(i)==0);
                index_test = find(cnew.test(i));
                mr_train = mr(index_train);
                matrix_all_train = matrix_all(index_train,:);
                [~,P_Pearson] = corrcoef([mr_train matrix_all_train]);
                [~,P_Kendall] = corr([mr_train matrix_all_train],'Type','Kendall');
                [~, P_Sperman ] = corr([mr_train matrix_all_train],'Type','Spearman');
                index_Pearson = find(P_Pearson(2:end,1) < 0.1);
                index_Kendall = find(P_Kendall(2:end,1) < 0.1);
                index_Spearman = find(P_Sperman(2:end,1) < 0.1);
                index_all_reduced = unique([index_Pearson' index_Kendall' index_Spearman']);
                matrix_all_train_reduced = matrix_all_train(:,index_all_reduced);
                mean_matrix_all_train = mean(matrix_all_train_reduced);
                std_matrix_all_train = std(matrix_all_train_reduced);
                matrix_all_train_norm = (matrix_all_train_reduced - mean_matrix_all_train)./std_matrix_all_train;
                mr_test = mr(index_test);
                matrix_all_test = matrix_all(index_test,:);
                matrix_all_test_reduced = matrix_all_test(:,index_all_reduced);
                matrix_all_test_norm = (matrix_all_test_reduced - mean_matrix_all_train)./std_matrix_all_train;
                t = templateTree('MaxNumSplits',NumSplits,'MinLeafSize',LeafSize);
                MdlOpt_train = fitrensemble(matrix_all_train_norm,mr_train,'NumLearningCycles',NTrees,'LearnRate',Lrate,'Learners',t);   
                mrhat_train = predict(MdlOpt_train,matrix_all_train_norm);
                MSE_train_norm = sum((mr_train - mrhat_train).^2)/sum((mr_train - mean(mr_train)).^2);
                mrhat_test = predict(MdlOpt_train,matrix_all_test_norm);
                MSE_test_norm = sum((mr_test - mrhat_test).^2)/sum((mr_test - mean(mr_test)).^2);
                SE_vector_test = [SE_vector_test MSE_test_norm];
                SE_vector_train = [SE_vector_train MSE_train_norm]; 
        end
    end
    MSE_train = mean(SE_vector_train);
    MSE_test =  mean(SE_vector_test);
    sterr_MSE_train = 2*std(SE_vector_train)/sqrt(length(SE_vector_train));
    sterr_MSE_test = 2*std(SE_vector_test)/sqrt(length( SE_vector_test));
    MSE_alpha_lambda_mat = [MSE_alpha_lambda_mat [NumSplits;LeafSize;Lrate;NTrees;MSE_train;sterr_MSE_train;MSE_test;sterr_MSE_test]];
end
% Min MSE model parameters:
[MSE_test_min,index_min] = min(MSE_alpha_lambda_mat(7,:));  
NumSplits_min = MSE_alpha_lambda_mat(1,index_min);   
LeafSize_min = MSE_alpha_lambda_mat(2,index_min); 
Lrate_min = MSE_alpha_lambda_mat(3,index_min); 
Ntrees_min = MSE_alpha_lambda_mat(4,index_min); 
MSE_train_min = MSE_alpha_lambda_mat(5,index_min);    
sterr_MSE_train_min = MSE_alpha_lambda_mat(6,index_min);      
sterr_MSE_test_min = MSE_alpha_lambda_mat(8,index_min );   
% Select variables for final GB model: 
[R_Pearson,P_Pearson] = corrcoef([mr matrix_all]);
[R_Kendall,P_Kendall] = corr([mr matrix_all] , 'Type' , 'Kendall');
[R_Sperman,P_Sperman] = corr([mr matrix_all] , 'Type' , 'Spearman');
index_Pearson = find( P_Pearson( 2:end , 1 ) < 0.1 );
index_Kendall = find( P_Kendall( 2:end , 1 ) < 0.1 );
index_Spearman = find( P_Sperman( 2:end , 1 ) < 0.1 );
index_all_reduced = unique([index_Pearson' index_Kendall' index_Spearman']);
Varnames_reduced = Varnames(index_all_reduced);                   
matrix_all_reduced = matrix_all(:,index_all_reduced);
% Standardization:
mean_matrix_all = mean(matrix_all_reduced);
std_matrix_all = std(matrix_all_reduced);       
matrix_all_norm = (matrix_all_reduced - mean_matrix_all)./std_matrix_all;
% Final GB regression:
t_all = templateTree('MaxNumSplits',NumSplits_min,'MinLeafSize',LeafSize_min);
MdlOpt_all = fitrensemble(matrix_all_norm,mr,'LearnRate',Lrate_min,'NumLearningCycles',Ntrees_min,'Learners',t_all);
mrhat = predict(MdlOpt_all,matrix_all_norm);
R_square = 1 - (sum((mr - mrhat).^2 )/sum((mr - mean(mr)).^2));

% Display results:
fprintf('\n\n <strong>Gradient Boost regression results </strong>\n')
fprintf('min MSE model parameters: \nNtrees = %d \nNSplits = %d\nLeafSize = %d \nLearnRate = %.4f',...
    Ntrees_min,NumSplits_min,LeafSize_min, Lrate_min) ; 
fprintf('minimal MSE (train set)= %d , with SE = %d \n',MSE_train_min,sterr_MSE_train_min)
fprintf('minimal MSE (test set)= %d , with SE = %d \n',MSE_test_min,sterr_MSE_test_min)
fprintf('R_square = %.4f\n', R_square); 
imp = predictorImportance( MdlOpt_all ); 
BFC = [125/255 131/255 128/255]; 
BEC = [23/255 28/255 233/255]; 
figure
cat = categorical(Varnames_reduced); 
cat = reordercats(cat, Varnames_reduced);
bar(cat,imp,'FaceColor', BFC,'EdgeColor',BEC,'LineWidth',1.5);
yline(mean(imp),'--k','LineWidth',2)
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
% Save the results:
save GB_5fold_repart_V1_mr.mat  MdlOpt_all  imp  MSE_alpha_lambda_mat  matrix_all  mr  Varnames_reduced

%% Hyperparameter optimization: 
upperlimit = MSE_test_min + sterr_MSE_test_min ; 
opt_ind = find(MSE_alpha_lambda_mat( 7 , : ) < upperlimit);
figure
subplot(2,2,1)
histogram(MSE_alpha_lambda_mat( 1 , opt_ind ))
xlabel('Maximal number of splits')
subplot(2,2,2)
histogram(MSE_alpha_lambda_mat( 2 , opt_ind ))
xlabel('Minimal number of leaves')
subplot(2,2,3)
histogram(MSE_alpha_lambda_mat( 4 , opt_ind ))
xlabel('Numbar of trees')
subplot(2,2,4)
histogram(MSE_alpha_lambda_mat( 3 , opt_ind ))
xlabel('Learn Rate')