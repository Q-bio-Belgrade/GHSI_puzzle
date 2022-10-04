%% GHSI data transformation and initial analysis
% m/r dataset (85 countries)
% 8 GHS Index categories + 18 demographic variables
% Used functions:
% substituteoutlier

clear

%% Load Data

load mr_GHSI_demo_data.mat

%% Data transformation

mr_GHSI_demo_data_transf(:,1) = log(mr_GHSI_demo_data(:,1)) ; % m/r
mr_GHSI_demo_data_transf(:,2) = (mr_GHSI_demo_data(:,2)).^(1/3) ; % Overall 
mr_GHSI_demo_data_transf(:,3) = mr_GHSI_demo_data(:,3) ; % Prevent 
mr_GHSI_demo_data_transf(:,4) = mr_GHSI_demo_data(:,4) ; % Detect
mr_GHSI_demo_data_transf(:,5) = (mr_GHSI_demo_data(:,5)).^(1/3) ; % Respond 
mr_GHSI_demo_data_transf(:,6) = sqrt(mr_GHSI_demo_data(:,6)) ; % Health
mr_GHSI_demo_data_transf(:,7) = sqrt(mr_GHSI_demo_data(:,7)) ; % Norms
mr_GHSI_demo_data_transf(:,8) = (mr_GHSI_demo_data(:,8)).^2 ; % Risk
mr_GHSI_demo_data_transf(:,9) = sqrt(mr_GHSI_demo_data(:,9)) ; % Covid
mr_GHSI_demo_data_transf(:,10) = sqrt(mr_GHSI_demo_data(:,10)) ; % PD
mr_GHSI_demo_data_transf(:,11) = (mr_GHSI_demo_data(:,11)).^2 ; % UP
mr_GHSI_demo_data_transf(:,12) = mr_GHSI_demo_data(:,12) ; % MA
mr_GHSI_demo_data_transf(:,13) = log(mr_GHSI_demo_data(:,13)) ; % IM
mr_GHSI_demo_data_transf(:,14) = log(mr_GHSI_demo_data(:,14)) ; % GDP
mr_GHSI_demo_data_transf(:,15) = - sqrt(max(mr_GHSI_demo_data(:,15))-mr_GHSI_demo_data(:,15)) ;  % HDI
mr_GHSI_demo_data_transf(:,16) = - sqrt(max(mr_GHSI_demo_data(:,16))-mr_GHSI_demo_data(:,16)) ;  % I-E
mr_GHSI_demo_data_transf(:,17) = log(mr_GHSI_demo_data(:,17)) ; % RE
mr_GHSI_demo_data_transf(:,18) = - sqrt(max(mr_GHSI_demo_data(:,18))-mr_GHSI_demo_data(:,18)) ;  % CH
mr_GHSI_demo_data_transf(:,19) = mr_GHSI_demo_data(:,19) ; % AL
mr_GHSI_demo_data_transf(:,20) = - log(max(mr_GHSI_demo_data(:,20))-mr_GHSI_demo_data(:,20)) ;  % OB
mr_GHSI_demo_data_transf(:,21) = mr_GHSI_demo_data(:,21) ; % SM
mr_GHSI_demo_data_transf(:,22) = (mr_GHSI_demo_data(:,22)).^(1/3) ; % CD
mr_GHSI_demo_data_transf(:,23) = sqrt(mr_GHSI_demo_data(:,23)) ; % RBP
mr_GHSI_demo_data_transf(:,24) = - sqrt(max(mr_GHSI_demo_data(:,24))-mr_GHSI_demo_data(:,24)) ;  % IN
mr_GHSI_demo_data_transf(:,25) = - sqrt(max(mr_GHSI_demo_data(:,25))-mr_GHSI_demo_data(:,25)) ;  % BCG
mr_GHSI_demo_data_transf(:,26) = log(mr_GHSI_demo_data(:,26)) ; % ON
mr_GHSI_demo_data_transf(:,27) = log(mr_GHSI_demo_data(:,27)) ; % PL

% Remove the outliers: 
[mr_GHSI_demo_data_transf_out,out_ind] = substituteoutlier(mr_GHSI_demo_data_transf);
% Check skewness of transformed data:
skewness(mr_GHSI_demo_data_transf_out)
% Display number of removed outliers: 
disp(sum(out_ind))

% Save transformed data: 
save mr_GHSI_demo_data_transf_out mr_GHSI_demo_data_transf_out mr_GHSI_demo_varnames

%% Save table as *.xlsx file: 
mr_GHSI_tr_out_tab = array2table(mr_GHSI_demo_data_transf_out) ; 
writetable(mr_GHSI_tr_out_tab,'mr_transf_out_data.xlsx','Sheet',3) ; 

%% Creatre correlation matrix of all variables: 
[R,P] = corrcoef(mr_GHSI_demo_data_transf_out) ; 
mr_variables_corrcoef_Table = array2table(R,'VariableNames', mr_GHSI_demo_varnames, 'RowNames', mr_GHSI_demo_varnames)  
mr_variables_pVal_Table = array2table(P,'VariableNames', mr_GHSI_demo_varnames, 'RowNames', mr_GHSI_demo_varnames)  

% save the results and write to excel: 
save mr_correlation_tables mr_variables_corrcoef_Table mr_variables_pVal_Table
writetable(mr_variables_corrcoef_Table, 'Corrcoefs.xlsx','Sheet',1) ; 
writetable(mr_variables_pVal_Table, 'Corrcoefs.xlsx','Sheet',2)

%% Correlation scatterplots of m/r and selected GHSI categories: 

clear
load mr_GHSI_demo_data_transf_out

mr = mr_GHSI_demo_data_transf_out(:,1) ; 
index = mr_GHSI_demo_data_transf_out(:,[4,5,6,9]) ; 
mr_varname = mr_GHSI_demo_varnames(1) ; 
index_varname = mr_GHSI_demo_varnames([4,5,6,9]) ; 

[R_all, P_all] = corrcoef([mr index]) ; 
R = R_all(2:end, 1) ; 
P = P_all(2:end,1) ; 
for i = 1:4
    p = polyfit(index(:,i),mr,1);
    h = min(index(:,i)):0.1:max(index(:,i)) ; 
    pv = polyval(p,h);
    figure
    plot(index(:,i),mr,'o','MarkerSize',5,'MarkerEdgeColor',[17/255 11/255 204/255],...
        'MarkerFaceColor',[17/255 11/255 204/255])
    hold on
    plot(h,pv,'r--','LineWidth',2)
    xlabel(index_varname(i),'FontSize' ,16);
    ylabel('m/r','FontSize' ,16);
    str = {['R = ',num2str(R(i))],['P = ',num2str(P(i))]} ; 
    annotation('textbox',[0.15,0.8,0.1,0.1],'String',str)
    hold off
end

