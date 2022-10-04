function [B_fin , FitInfo_fin, MSE_lambda_mat, new_x, new_varnames ] = relaxedlasso_max_sparse(x,y,x_names)
% LASSO (Least Absolute Shrinkage and Selection Operator)
% function, adopted for COVID-19 GHSI research (Kfold cv n_fold = 5, n_repart =40)
% Two rounds of lasso(Relaxed) with sparsest model selected in both rounds.
% Input:
% x - predictor matrix
% y - response variable
% x_names - predictor names

% standardize input variables: 
norm_x = normalize(x); 
% Scan for Lambda values:
[ ~ , FitInfo_scan ] = lasso( norm_x , y );
lambda_vec = FitInfo_scan.Lambda;
% CV parameters:
n_fold = 5;
n_repart = 40;
cv = cvpartition(y,'KFold',n_fold, 'Stratify',false);

MSE_lambda_mat = [];
lena_param = length( lambda_vec );
param_index_perm_vec = randperm( lena_param );

for index = param_index_perm_vec
    lambda = lambda_vec(index);
    SE_vector_train = [];
    SE_vector_test = [];
    for repart = 1:n_repart 
        cnew = repartition(cv);
        parfor  i = 1:n_fold
                % Form and standardize test and train sets:
                index_train = find( cnew.test(i)==0);
                index_test = find( cnew.test(i));
                y_train = y(index_train );
                x_train = x(index_train,:);
                mean_x_train = mean(x_train);
                std_x_train = std(x_train);
                x_train_norm = (x_train - mean_x_train)./std_x_train;
                y_test = y(index_test);
                x_test = x(index_test,:);
                x_test_norm = (x_test - mean_x_train)./std_x_train;
                % Lasso regression on train set:
                [B_train,FitInfo_train] = lasso(x_train_norm,y_train,...
                    'Lambda',lambda,'PredictorNames',x_names);
                coef_train = B_train;
                coef0_train = FitInfo_train.Intercept;
                yhat_train = x_train_norm*coef_train + coef0_train;
                % MSE on train set:
                MSE_train_norm = sum((y_train - yhat_train).^2)/sum((y_train - mean(y_train)).^2);
                yhat_test = x_test_norm*coef_train + coef0_train ;
                % MSE on test set:
                MSE_test_norm = sum((y_test - yhat_test).^2)/sum((y_test - mean(y_test)).^2);
                SE_vector_train = [SE_vector_train MSE_train_norm];
                SE_vector_test = [SE_vector_test MSE_test_norm];
        end
    end
    % MSE for given lambda value:
    MSE_train = mean(SE_vector_train);
    MSE_test =  mean(SE_vector_test);
    sterr_MSE_train = 2*std(SE_vector_train)/sqrt(length(SE_vector_train));
    sterr_MSE_test = 2*std(SE_vector_test)/sqrt(length(SE_vector_test));
    % write the results:
    MSE_lambda_mat = [MSE_lambda_mat [lambda;MSE_train;sterr_MSE_train;MSE_test;sterr_MSE_test]];

end

% find min MSE model:
[MSE_test_min,index_min] = min(MSE_lambda_mat(4,:));  
sterr_MSE_test_min =  MSE_lambda_mat( 5 , index_min ); 
lambda_min = MSE_lambda_mat( 1 , index_min ); 
MSE_test_min_err = MSE_test_min + sterr_MSE_test_min;

% Find models within 1 sterr from min MSE:
index_min_all =  find(MSE_lambda_mat(4,:) <  MSE_test_min_err);
lambda_min_all =  MSE_lambda_mat(1, index_min_all );
% Find sparsest model (highest lambda value) among them:
lambda_min_all_max = max( lambda_min_all );
% Use that lambda in Lasso regression on the entire dataset:
[B,FitInfo] = lasso(norm_x,y,'Lambda',lambda_min_all_max,'PredictorNames',x_names);
% Keep only variables selected in this model:
KeepIndex = find(B(:)~=0); 
x = x(:,KeepIndex) ; 
x_names = x_names(KeepIndex) ; 

% REPEAT Lasso (Round 2): 
[~,FitInfo_scan] = lasso(normalize(x),y);
lambda_vec = FitInfo_scan.Lambda;
n_fold = 5;
n_repart = 40;
cv = cvpartition(y,'KFold',n_fold,'Stratify',false); 

MSE_lambda_mat = [];

lena_param = length( lambda_vec );
param_index_perm_vec = randperm( lena_param );

for index = param_index_perm_vec
    lambda = lambda_vec(index);
    SE_vector_train = [];
    SE_vector_test = [];
    for repart = 1:n_repart 
        cnew = repartition(cv);
        parfor  i = 1:n_fold
                index_train = find(cnew.test(i)==0);
                index_test = find(cnew.test(i));
                y_train = y(index_train);
                x_train = x(index_train,:);
                mean_x_train = mean(x_train);
                std_x_train = std(x_train);
                x_train_norm = (x_train - mean_x_train)./std_x_train;
                y_test = y(index_test);
                x_test = x(index_test,:);
                x_test_norm =(x_test - mean_x_train)./std_x_train;
                [B_train,FitInfo_train] = lasso(x_train_norm,y_train,'Lambda',lambda,'PredictorNames',x_names);
                coef_train = B_train;
                coef0_train = FitInfo_train.Intercept;
                yhat_train = x_train_norm*coef_train + coef0_train;
                MSE_train_norm = sum((y_train - yhat_train).^2)/sum((y_train - mean(y_train)).^2);
                yhat_test = x_test_norm*coef_train + coef0_train;
                MSE_test_norm = sum((y_test - yhat_test).^2)/sum(( y_test - mean(y_test)).^2);
                SE_vector_train = [SE_vector_train MSE_train_norm];
                SE_vector_test = [SE_vector_test MSE_test_norm];
        end
    end
    MSE_train = mean(SE_vector_train);
    MSE_test =  mean(SE_vector_test);
    sterr_MSE_train = 2*std(SE_vector_train)/sqrt(length(SE_vector_train));
    sterr_MSE_test = 2*std(SE_vector_test)/sqrt( length(SE_vector_test));
    MSE_lambda_mat = [ MSE_lambda_mat [lambda;MSE_train;sterr_MSE_train;MSE_test;sterr_MSE_test]];
end
% Find min MSE model:
[MSE_test_min,index_min] = min(MSE_lambda_mat(4,:));      
sterr_MSE_test_min = MSE_lambda_mat(5,index_min); 
lambda_min = MSE_lambda_mat(1,index_min);
% Find sparsest model:
MSE_test_min_err = MSE_test_min + sterr_MSE_test_min;
index_min_all = find(MSE_lambda_mat(4,:) < MSE_test_min_err);
lambda_min_all = MSE_lambda_mat(1,index_min_all);
lambda_min_all_median = median(lambda_min_all);
lambda_min_all_max = max(lambda_min_all);
% Lasso regression with sparsest lambda value:
norm_x = normalize(x); 
[B_fin,FitInfo_fin] = lasso(norm_x,y,'Lambda',lambda_min_all_max,'PredictorNames', x_names);
ModelPredictors = FitInfo_fin.PredictorNames(B_fin(:)~=0); 
coef = B_fin(B_fin(:)~=0);
coef0 = FitInfo.Intercept;
yhat = norm_x*B_fin + coef0;
R_square = 1 - (sum((y - yhat).^2 )/sum((y - mean(y)).^2 )); 

% results table
Koeficijenti_tabela = [coef0; coef]; 
Prediktori_tabela = [{'Intercept'}, ModelPredictors]; 
LASSO_Results_fin = array2table(Koeficijenti_tabela, 'RowNames',Prediktori_tabela, 'VariableNames', {'Estimate'}) ;
disp(LASSO_Results_fin)
fprintf('lambda = %.2f \nmin MSE = %.4f , SE min MSE = %.6f\n',...
lambda_min,MSE_test_min, sterr_MSE_test_min) ; 
fprintf('R_square = %.6f \n', R_square) ; 
% Bar graph:
cat = categorical(ModelPredictors); 
cat = reordercats(cat, ModelPredictors);
BFC = [125/255 131/255 128/255]; 
BEC = [23/255 28/255 233/255]; 
figure
bar(cat, coef,'FaceColor', BFC,'EdgeColor',BEC,'LineWidth',1.5);
ylabel('Model coefficients')
title('Relaxed Lasso regression (sparse)')

new_x = x ; 
new_varnames = x_names ; 

end

