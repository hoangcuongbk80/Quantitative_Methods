function [x y] = RemoveData(x,y)
% Removes any rows in x and y that contain any NaN or Inf values. Use the 
% functions isnan, find, and removerows. Remember that you have to remove 
% the same rows in both x and y if you find any bad values.
% ====================== YOUR CODE HERE ======================
% x = ...
% y = ...
RowsToBeRemoved = find(sum((isnan([x y])),2)+sum(isinf([x y]),2)~=0);
%x = removerows(x, RowsToBeRemoved);
%y = removerows(y, RowsToBeRemoved);
x(RowsToBeRemoved,:)=[];
y(RowsToBeRemoved,:)=[];
% ============================================================

end