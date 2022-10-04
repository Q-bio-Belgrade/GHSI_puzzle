function  [ matrix_all_out , index_out_var_mat  ]  =  substituteoutlier( matrix_all )

num_variable = size( matrix_all , 2 )    ;

index_out_var_mat = []   ;

matrix_all_out = []      ;

for i=1:num_variable
    
    variable = matrix_all( : , i )  ;
    
    median_variable = nanmedian( variable )  ;
    
    index_out_var = find( isoutlier( variable )==1 )   ; 
    
    variable_out = variable   ;
    
    variable_out( index_out_var ) =  median_variable   ;
    
    index_out_var_mat = [  index_out_var_mat  isoutlier( variable )  ]  ;
    
    matrix_all_out = [  matrix_all_out  variable_out   ]      ;
    
end