function [ta,t] = sdol_tsa(x,sr,tach,ppr,Nr)
% =========================================================================
% This code is programmed by System Design Optimization Lab (SDOL) at Korea
% Aerospace University (KAU)
% ============================== Input ====================================
% x: Vibration signal
% sr: Sampling frequency
% tach: Tachometer signal
% ppr: The number of pulses per 1 revolution
% Nr: The number of rotation to calculate TSA
% ============================== Output ===================================
% ta: Time synchronous average of signal x
% t: Sample time correspond to ta
% =========================================================================
if nargin == 4
    Nr = 1;
end
x = x(:); t = (0:1/sr:(length(x)-1)/sr)';
ppr = ppr*Nr;
if rem(length(tach),ppr)
    nrev = floor(length(tach)/ppr);
else
    nrev = length(tach)/ppr-1;
end

T = nan(nrev,1);
for j = 1:nrev
    T(j) = tach(ppr*j+1);
end
T = [tach(1); T]; T(T>t(end)) = [];

nrev = length(T)-1; N = round(min(diff(T),[],1)*sr); 
resamp = nan(nrev,N);
for j = 2:nrev+1
    tmp = (T(j)-T(j-1))/N; tmp = tmp*(0:N-1);
    resamp(j-1,:) = T(j-1)+tmp;
end
resx = interp1(t,x,resamp,'linear');
ta = mean(resx)';
f = 1/min(diff(T)); sr_resamp = N*f;
t = (0:N-1)'/sr_resamp;

end