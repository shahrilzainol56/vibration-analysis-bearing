function [ardata,p] = AR_filter(x,maxK,ix)
% ============================== Input ====================================
% x: Vibration signal
% maxK: The maximum filter order to calculate iterations
% ix: Plot the figure when the value is entered
% ============================== Output ===================================
% ardata: Residual signal
% p: Selected filter order
% =========================================================================
pp = 1:10:maxK; N = length(pp);
for ix=1:N
    a1 = aryule(x,pp(ix));                  % AR filter parameter
    xp = filter([0 -a1(2:end)],1,x);        
    xn = x - xp;
    k(ix) = kurtosis(xn);
end
ind = find(k == max(k),1); p = pp(ind);
a1 = aryule(x,p); xp = filter([0 -a1(2:end)],1,x);
ardata = x - xp;                            % Residual signal
% Plot
if nargin == 3
    figure
    plot(pp,k)
    hold on
    plot(p,k(ind),'MarkerSize',10)
    xlabel('Order(p)'); ylabel('Kurtosis');
end
end