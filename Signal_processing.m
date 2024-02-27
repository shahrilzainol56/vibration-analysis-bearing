set(0,'defaultAxesFontSize',15); set(0,'defaultLineLinewidth',1.5);        
set(0,'DefaultTextInterpreter','none');   
% ========================= Explanation ===================================
% This code is intended to show the signal processing in the paper
% step-by-step using KAU bearing
% Used data is 'bearing1.mat' from Diagnostics101_data.zip
% https://www.kau-sdol.com/bearing
% 1. AR filter
% 2. Spectral kurtosis
% 3. Envelope analysis
% 4. Empirical Mode Decomposition
% =========================================================================
%% 1. AR filter
clear; clc; close all;
currentFolder = pwd;
addpath(currentFolder);
addpath('Functions');                                   % Input function folder directory
cd('Data_repository\KAU_bearing');                      % Input data folder directory
x = load('bearing1.mat'); x = x.vib; N = length(x);     % Load data
fs = 51.2e3; t = [0:N-1]/fs; fr = 1200/60;              % Sampling frequency(Hz), time vector(s), rotating frequency(Hz)
bff = [4.4423 6.5577 0.4038 5.0079]*fr;                 % Bearing fault frequencies [BPFO BPFI BSF FTF]
[xr,p] = AR_filter(x,700);                              % AR filtering
k1 = kurtosis(x); k2 = kurtosis(xr);                    % Calculate kurtosis
% Time domain signal <Figure 4(a)>
figure(1);
subplot(211); plot(t,x); ylim([-1.5 1.5]);
title(['Raw signal (Kurtosis: ',num2str(k1),')']); 
subplot(212); plot(t,xr); ylim([-1.5 1.5]);
title(['AR filtered signal (Kurtosis: ',num2str(k2),')']); 
xlabel('Time (s)');

% Frequency domain signal <Figure 4(b)>
f = (0:N-1)'/N*fs; f = f(1:ceil(N/2));
X1 = abs(fft(abs(hilbert(x))))/N; X1 = X1(1:ceil(N/2));
X2 = abs(fft(abs(hilbert(xr))))/N; X2 = X2(1:ceil(N/2));
figure(2);
subplot(211); plot(f,X1); xlim([0 200]); ylim([0 0.02]);
title(['Envelope spectrum (Raw, Kurtosis: ',num2str(k1),')']);
xline(bff(1)-fr,'k--','linewidth',2); xline(bff(1),'r--','linewidth',2); 
xline(bff(1)+fr,'k--','linewidth',2);
subplot(212); plot(f,X2); xlim([0 200]); ylim([0 0.02]);
title(['Envelope spectrum (AR filtered, Kurtosis: ',num2str(k2),')']);
xline(bff(1)-fr,'k--','linewidth',2); xline(bff(1),'r--','linewidth',2); 
xline(bff(1)+fr,'k--','linewidth',2);
xlabel('Frequency (Hz)');

%% 2. Spectral kurtosis
k = kurtosis(x);
% Kurtogram <Figure 5(a)>
figure; kurtogram(xr,fs);
cf = 20.8e3; bandwidth = 3.2e3;                 % Center frequency & bandwidth obtained from kurtogram
xx1 = bandpass(xr,[cf-bandwidth/2 cf+bandwidth/2],fs); k1 = kurtosis(xx1);
xx2 = bandpass(xr,[100 7000],fs); k2 = kurtosis(xx2); k2 = kurtosis(xx2);

% Time domain signal <Figure 5(b)>
figure;
subplot(311); plot(t,xr); title(['AR filterd signal (Kurtosis: ',num2str(kurtosis(xr1)),')']);
subplot(312); plot(t,xx1);
title(['SK Filtered signal, ',num2str(cf-bandwidth/2),' ~ ',num2str(cf+bandwidth/2),...
    'Hz (Kurtosis: ',num2str(k1),')']);
subplot(313); plot(t,xx2);
title(['SK Filtered signal, ',num2str(100),' ~ ',num2str(7000),'Hz (Kurtosis: ',num2str(k2),')']);
xlabel('Time (s)');

% Frequency domain signal <Figure 5(c)>
X = abs(fft(xr))/N; X = X(1:ceil(N/2));
X1 = abs(fft(xx1))/N; X1 = X1(1:ceil(N/2));
X2 = abs(fft(xx2))/N; X2 = X2(1:ceil(N/2));
figure;
subplot(311); stem(f,X); title(['AR filterd signal (Kurtosis: ',num2str(kurtosis(xr1)),')']);
subplot(312); stem(f,X1);
title(['SK Filtered signal, ',num2str(cf-bandwidth/2),' ~ ',num2str(cf+bandwidth/2),...
    'Hz (Kurtosis: ',num2str(k1),')']);
subplot(313); plot(f,X2);
title(['SK Filtered signal, ',num2str(100),' ~ ',num2str(7000),'Hz (Kurtosis: ',num2str(k2),')']);
xlabel('Frequency (Hz)');

%% 3. Envelope analysis
% Bandpass filtered signal <Figure 6(a)>
figure;
plot(t,xx1); xlim([0 0.3]); xticks([]); yticks([]);
ylabel('Bandpassed signal');

% FFT of signal (a) <Figure 6(b)>
figure;
stem(f,X1); xticks([0 200]); xticklabels({'0','200'}); xlim([0 200]);
xline(bff(1),'g--','linewidth',2); xline(bff(2),'r--','linewidth',2); 
xline(bff(3),'b--','linewidth',2); xline(bff(4),'c--','linewidth',2);
legend('','BPFO', 'BPFI', 'BSF', 'FTF')
yticks([]); ylim([0 2e-7]); ylabel('Amplitude');
xn = abs(hilbert(xx1));

% Scale-up view in(0,200) Hz of (b) <Figure 6(c)>
figure; hold on;
plot(t,xx1,'color',[0.5 0.5 0.5]); plot(t,xn,'r');
xlim([0 0.3]); xticks([]); yticks([]);
ylabel('Envelope signal');

% envelope of signal (a) <Figure 6(d)>
X2 = abs(fft(xn-mean(xn)))/N; X2 = X2(1:ceil(N/2));
figure;
stem(f,X2); xticks([0 200]); xticklabels({'0','200'}); xlim([0 200]);
xline(bff(1),'g--','linewidth',2); xline(bff(2),'r--','linewidth',2); 
xline(bff(3),'b--','linewidth',2); xline(bff(4),'c--','linewidth',2);
legend('','BPFO', 'BPFI', 'BSF', 'FTF')
yticks([]); ylabel('Amplitude');

% FFT of envelope signal (d) <Figure 6(e)>
X2 = abs(fft(xn-mean(xn)))/N; X2 = X2(1:ceil(N/2));
figure;
stem(f,X2); xticks([0 2.5e4]); xticklabels({'0','25k'}); xlim([0 2.5e4]);
yticks([]); ylabel('Amplitude');

% Scale-up view in(0,200)Hz of (e) <Figure 6(f)>
figure;
stem(f,X2); xticks([0 2.5e4]); xticklabels({'0','200'}); xlim([0 200]);
p1 = xline(bff(2),'r--','linewidth',1.5);
p2 = xline(bff(2)-fr,'k--','linewidth',1.5); xline(bff(2)+fr,'k--','linewidth',1.5);
legend([p1,p2],'f_{bpfi}','f_{bpfi}\pmf_r','Interpreter','tex');
yticks([]); ylabel('Amplitude');
%% 4. Empirical Mode Decomposition
[imf,res] = emd(x);                                 % EMD
for ix = 1:10
    tmp = abs(fft(imf(:,ix))); tmp = tmp(1:ceil(N/2));
    XX(:,ix) = tmp/length(imf(:,ix));
end
R = abs(fft(res)); R = R(1:ceil(N/2));

% Time domain signal <Figure 7(a)>
figure;
subplot(611); plot(t,x); ylabel('Raw'); title('Time domain IMF');
subplot(612); plot(t,imf(:,1)); ylabel('IMF1');
subplot(613); plot(t,imf(:,2)); ylabel('IMF2');
subplot(615); plot(t,imf(:,10)); ylabel('IMF10');
subplot(616); plot(t,res); ylabel('Residual'); xlabel('Time (s)');
set(gcf,'Position',[-1389 76 684 844]);

% Frequency domain signal <Figure 7(b)>
figure;
subplot(611); stem(f,X); ylabel('Raw'); title('Frequency domain IMF');
subplot(612); stem(f,XX(:,1)); ylabel('IMF1');
subplot(613); stem(f,XX(:,2)); ylabel('IMF2');
subplot(615); stem(f,XX(:,10)); ylabel('IMF10');
subplot(616); stem(f,R); ylabel('Residual'); xlabel('Frequency (Hz)');
set(gcf,'Position',[-1389 76 684 844]);

