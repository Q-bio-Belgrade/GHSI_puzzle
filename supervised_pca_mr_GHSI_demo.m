%% Supervised PCA m/r GHSI dataset 

clear
rng default
load mr_GHSI_demo_data_transf_out

mr = mr_GHSI_demo_data_transf_out(:,1); 
matrix_all = mr_GHSI_demo_data_transf_out(:,[4,5,6,8,10:end]); 
Varnames = mr_GHSI_demo_varnames([4,5,6,8,10:end]); 

% Correlation of variables with m/r: 
[R_all,P_all] = corrcoef([mr matrix_all]); 
R = R_all(2:end, 1); 
P = P_all(2:end, 1); 

% Matrix of hyperparametert theta (threshold) and m (num_PC):
param_matrix = [];  
for threshold = 0:0.05:1
    for num_PC = 1:10
        param_matrix = [param_matrix [threshold;num_PC]];        
    end
end

lena_param = size(param_matrix,2);
param_index_perm_vec = randperm(lena_param);

% Cross Validation parameters: 
n_fold = 5; 
n_repart = 40;
cv = cvpartition(mr,'KFold',n_fold,'Stratify',false);

Model_param_mat = []; 

for index = param_index_perm_vec 
    
    % theta and m compination selection
    theta = param_matrix(1,index); 
    m = param_matrix(2,index); 
    
    % Standard error vectors: 
    SE_vector_test = [];  
    SE_vector_train = [];
    
    % Select variables that satisfy R > theta :
    index_keep = abs(R) > theta ; 
    reduced_data = matrix_all(:,index_keep); 
    reduced_varnames = Varnames(index_keep);
    
    % Cross-validation loop: 
    for repart = 1:n_repart 

        cnew = repartition( cv );

        parfor  i = 1:n_fold

                mr_test = mr(cnew.test(i) == 1);
                mr_train = mr(cnew.test(i) == 0); 

                train_data = reduced_data(cnew.test(i)== 0,:); 
                train_data_norm = normalize(train_data); 

                test_data = reduced_data(cnew.test(i) == 1,:); 
                test_data_norm = (test_data - mean(train_data))./std(train_data); 

                % PCA on train set: 
                [coeff,score, ~,~,explained] = pca(train_data_norm) ; 

                if  m < size(score, 2)
                    pc_keep = 1:m ; 
                else
                    pc_keep = 1:size(score, 2) ; 
                end

                train_score = train_data_norm*coeff(:,pc_keep) ;  
                test_score = test_data_norm*coeff(:,pc_keep) ; 

                train_score_norm = normalize(train_score) ; 
                test_score_norm = (test_score - mean(train_score))./std(train_score) ; 

                mdl = fitlm(train_score_norm, mr_train) ;

                mrhat_train = feval(mdl,train_score_norm) ; 

                MSE_train_norm = sum((mr_train - mrhat_train).^2)/sum((mr_test - mean(mr_test)).^2) ; 

                SE_vector_train = [SE_vector_train, MSE_train_norm] ; 

                mrhat_test = feval(mdl,test_score_norm) ; 

                MSE_test_norm = sum(  ( mr_test  -  mrhat_test ).^2  )/sum(  ( mr_test  -  mean(mr_test) ).^2  )    ;

                SE_vector_test = [SE_vector_test, MSE_test_norm] ; 

        end  

    end
        
    MSE_test =  mean( SE_vector_test  )  ;

    sterr_MSE_test = 2*std( SE_vector_test )/sqrt( length( SE_vector_test ) )  ;

    MSE_train =  mean( SE_vector_train  )  ;

    sterr_MSE_train = 2*std( SE_vector_train )/sqrt( length( SE_vector_train ) )  ;

    Model_param_mat = [Model_param_mat [theta; m; MSE_train ;  sterr_MSE_train ; MSE_test; sterr_MSE_test]] ; 

end


[ MSE_test_min , index_min ]  = min(  Model_param_mat( 5 , : )  )  ; 

theta_min_MSE = Model_param_mat(1,index_min) ; 
m_min_MSE = Model_param_mat(2,index_min) ; 
sterr_min_MSE = Model_param_mat(6,index_min) ; 

save supervised_PCA_mr_GHSI_V1 Model_param_mat mr matrix_all Varnames R  

%% Min MSE model 

index_keep = abs(R) > theta_min_MSE ; 
reduced_data = matrix_all(:,index_keep) ; 
reduced_data_norm  = normalize(reduced_data) ; 
reduced_varnames = Varnames(index_keep) ; 
[coeff, score, ~,~,explained] = pca(reduced_data_norm) ; 
scores_keep = score(:,1:m_min_MSE) ;
fin_mdl = fitlm(scores_keep, mr) ;
disp(fin_mdl)
fprintf('MSE = %f\ntheta = %f\nm = %f\n', MSE_test_min,theta_min_MSE,m_min_MSE)

%% Correlation of the principal components with the variables entering PCA
    
Y = scores_keep  ; 
X = reduced_data ;

% Y variable name:

for i = 1:m_min_MSE
NaziviY(i) = {['PC' num2str(i)]};
end

% X variable names
NaziviX = reduced_varnames;  

% Bar face color:
BFC = [81/255 181/255 191/255]; 
% BFC = [235/255 196/255 99/255];

% bar edge color:
BEC = [0/255 0/255 0/255]; 

[nrowsX, ncolX] = size(X) ; 
[nrowsY, ncolY] = size(Y); 
X = substitutemissing(X) ; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nrowsX ~= nrowsY
    error('Number of rows in matrices X and Y must be the same!')
end   

if length(NaziviY)~= ncolY 
    error('NaziviY mora da sadrzi nazive za svaku kolonu Y')
end

if length(NaziviX)~= ncolX 
    error('NaziviX mora da sadrzi nazive za svaku kolonu X')
end

if any(BFC>1) || any(BFC <0) || any(BEC>1) || any(BFC <0)
    error('Boje uneti u RGB formatu, sa vrednostima od 0 do 1')
end

if (length(BFC) ~= 3) || (length(BEC) ~= 3)
    error('BFC i BEC moraju biti vektori dužine 3 (RGB)')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
barFontSize = 12; 
coeff = zeros(ncolX, ncolY); 
pVal = zeros(ncolX,ncolY); 
xTicks = categorical(NaziviX) ;
xTicks = reordercats(xTicks, NaziviX); 

for i = 1:ncolY
    
[R_all, P_all] = corrcoef([Y(:,i) X]); 
yosa = 'Pearson Correlation' ;
coeff = R_all(2:end,1); 
pVal(:,i) = P_all(2:end,1); 

Znacaj = {}; 
    for j = 1:length(pVal)
        if  pVal(j) <= 0.001
            Znacaj{j} = '***'; 
        elseif  pVal(j) <= 0.01
            Znacaj{j} = '**';
        elseif  pVal(j) <= 0.05
            Znacaj{j} = '*';
        else
            Znacaj{j} = 'ns';
        end
    end

    numberOfBars = ncolX;
    
    subplot(ncolY,1,i)
    for b = 1 : numberOfBars
        handleToThisBarSeries(b) = bar(xTicks(b),coeff(b), 'BarWidth', 0.7); %,
        set(handleToThisBarSeries(b), 'FaceColor', BFC,'EdgeColor',BEC,'LineWidth',1.5);
        barTopper = sprintf(Znacaj{b});
        if  coeff(b) >= 0 
            text(b, coeff(b), barTopper,'vert','bottom','horiz','center','FontSize',barFontSize); % xTicks(b)-0.2, coeff(b)+3,
        else
            text(b, coeff(b), barTopper,'vert','top','horiz','center','FontSize',barFontSize); % xTicks(b)-0.2, coeff(b)+3,
         
        end 
        hold on;
    end
    box off
    ylabel(yosa); 
    title(NaziviY{i});
end
