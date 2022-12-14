function pc_correlation_plots(scores,input_data,n, pc_name, input_varnames,r,c)
%pc_correlation_plots calculates the correlation of given principal components 
% with the variables entering PCA and returns rxc panel of correlation plots
%   scores - all PCs from PCA
%   input_data - data that entered pca
%   n - number of retained PCs
%   pc_name - name of PCs (e.g. "demo", "meteo", "age" ...) 
%   input_varnames - cell with names for each variable that entered PCA
%   r,c - rows and columns in panel

Y = scores(:,1:n)  ; 
X = input_data ;

for i = 1:n
NaziviY(i) = {[pc_name ' PC' num2str(i)]};
end

NaziviX = input_varnames;  

%  BarFaceColor:
BFC = [81/255 181/255 191/255]; 
% BFC = [235/255 196/255 99/255];

% BarEdgeColor:
BEC = [0/255 0/255 0/255]; 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nrowsX, ncolX] = size(X) ; 
[nrowsY, ncolY] = size(Y); 
X = substitutemissing(X) ; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Error check %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nrowsX ~= nrowsY
    error('Broj redova u matricama X i Y mora biti isti!')
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
        if pVal(j,i) <= 0.001
            Znacaj{j} = '***'; 
        elseif pVal(j,i) <= 0.01
            Znacaj{j} = '**';
        elseif pVal(j,i) <= 0.05
            Znacaj{j} = '*';
        else
            Znacaj{j} = 'ns';
        end
    end

    numberOfBars = ncolX;
   
    
    subplot(r,c,i)
    for b = 1 : numberOfBars
        % Plot one single bar as a separate bar series.
        handleToThisBarSeries(b) = bar(xTicks(b),coeff(b), 'BarWidth', 0.7); %,
        % Apply the color to this bar series
        set(handleToThisBarSeries(b), 'FaceColor', BFC,'EdgeColor',BEC,'LineWidth',1.5);
        % Place text atop the bar
    	barTopper = sprintf(Znacaj{b});
        if coeff(b) >= 0 
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
end

