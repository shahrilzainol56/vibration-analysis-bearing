function [feature, feature_name] = TimeFeatures(x)
% =========================================================================
% This code is programmed by System Design Optimization Lab (SDOL) at Korea
% Aerospace University (KAU)
% ============================== Input ====================================
% x : Input data
% ============================== Output ===================================
% feature : Calculated feature value
% feature_name : The name of features
% =========================================================================
x = x(:); N = length(x);
xm = mean(x);                                                   % 1. Mean
xsd = std(x);                                                   % 2. Standard deviation
xrms = rms(x);                                                  % 3. RMS
xsk = skewness(x);                                              % 4. Skewness
xkurt = kurtosis(x);                                            % 5. Kurtosis
xsf = xrms/(1/N*sum(abs(x)));                                   % 6. Shape factor
xcf = max(abs(x))/xrms;                                         % 7. Crest factor
xif = max(abs(x))/(1/N*sum(abs(x)));                            % 8. Impulse factor
xmf = max(abs(x))/((sum(sqrt(abs(x)))/length(x))^2);			% 9. Margin factor
xp = max(abs(x));                                               % 10. Peak
xp2p = max(x)-min(x);                                           % 11. Peak-to-peak
 
feature = [xm xsd xrms xsk xkurt xsf xcf xif xmf xp xp2p];
feature_name = {'MEAN','STD','RMS','SK','KUR','SF','CF','IF','MF','PEAK','P2P'};
end
