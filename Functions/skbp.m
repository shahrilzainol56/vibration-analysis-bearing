%%
function xb = skbp(x,fs)
% This code was made for bandpass filtering after SK
% Input
% x: Input signal, maybe AR filtered signal
% fs: Sampling frequency
% Output
% xb: Bandpass filtered signal
[~,~,~,fc,~,bw] = kurtogram(x,fs);
if fc-bw/2 == 0
    xb = lowpass(x,fc+bw/2,fs);
elseif fc+bw/2 == fs/2
    xb = highpass(x,fc-bw/2,fs);
else
    xb = bandpass(x,[fc-bw/2 fc+bw/2],fs);
end

end