function J3 = ScattMat(data,label)
% =========================================================================
% This code is programmed by Seokgoo Kim in 2018
% ============================= Input =====================================
% data(NxM) : N samples of M features
% label(Nx1) : Labels of N samples
% ============================= Output ====================================
% J3 : Calculated J3 value
% =========================================================================
label = label(:); class = unique(label);
M = length(class); N = length(label);
if size(data,1)~= length(label)
    data = data';
end
sw = 0; sb = 0;                             
mu = mean(data);                                                    % Global mean vector
for i = 1 : M
    temp = data(label==class(i),:);
    s = 1/(length(temp)-1)*(temp-mean(temp))'*(temp-mean(temp));    % Cov. matrix for class i
    sw = sw + length(temp)/N*s;                                     % Within-class scatter matrix
    sb = sb + length(temp)/N*(mean(temp)-mu)'*(mean(temp)-mu);      % Between class scatter matrix
end
sm = sw + sb;                                                       % Mixture scatter matrix
J3 = trace(inv(sw)*sm);                                     		% J3 value
end
