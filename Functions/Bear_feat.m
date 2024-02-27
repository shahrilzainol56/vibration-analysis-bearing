function [feature, feature_name] = Bear_feat(x, fs, bff, cutoff)
% =========================================================================
% This code is programmed by System Design Optimization Lab (SDOL) at Korea
% Aerospace University (KAU)
% ============================== Input ====================================
% x : Input data
% fs : Sampling frequency
% bff : Bearing fault frequency 1x4 matrix [bpfo,bpfi,ftf,bsf]
% cutoff : Bandwidth to find amplitude of fault frequency
% ============================== Output ===================================
% feature : Calculated feature value
% feature_name: The name of features
% =========================================================================
% FFT
x = abs(hilbert(x)); x = x-mean(x); N = length(x);
X = abs(fft(x))/N*2; X = X(1:ceil(N/2));
f = (0:N-1)'/N*fs; f = f(1:ceil(N/2));
% Find amplitude at bearing fault frequency
bpfo_ix = find(bff(1)-cutoff<f & f<bff(1)+cutoff); bpfo_amp = max(X(bpfo_ix,:));
bpfi_ix = find(bff(2)-cutoff<f & f<bff(2)+cutoff); bpfi_amp = max(X(bpfi_ix,:));
ftf_ix = find(bff(3)-cutoff<f & f<bff(3)+cutoff); ftf_amp = max(X(ftf_ix,:));
bsf_ix = find(bff(4)-cutoff<f & f<bff(4)+cutoff); bsf_amp = max(X(bsf_ix,:));
feature = [bpfo_amp, bpfi_amp, ftf_amp,bsf_amp];
feature_name = {'BPFO', 'BPFI', 'FTF','BSF'};
end
