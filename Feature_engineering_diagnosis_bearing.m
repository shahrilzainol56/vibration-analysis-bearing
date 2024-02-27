% cd('D:\Users\User\OneDrive - 한국항공대학교\논문\04_Tutorial_Python\02_Matlab Code\Practice');
set(0,'defaultAxesFontSize',15); set(0,'defaultLineLinewidth',1.5);        
set(0,'DefaultTextInterpreter','none');   
% ========================= Explanation ===================================
% This code describes the process from signal processing to feature
% selection using CWRU bearing data.
% Used data is CWRU bearing dataset available at the link below
% Note that each data was divided in 10 segments without overlapping
% https://csegroups.case.edu/bearingdatacenter/pages/download-data-file
% 1. Envelope analysis
% 2. Feature extraction (Time features, Features for bearing)
% 3. Scatter matrix (J3)
% =========================================================================
%% CWRU bearing
clear; clc; close all;
currentFolder = pwd;
addpath(currentFolder);
addpath('Functions');                                                       % Input function folder directory
cd('Data_repository/CWRU_bearing');                                         % Input data folder directory
bff = [3.5848 5.4152 0.3983 4.7135];
cutoff = 3; fr = 1730/60;
% =========================== Load data ===================================
load('Normal'); normal = x;
load('Outer'); outer = x;
load('Inner'); inner = x;

% 1. Envelope analysis
x1 = normal(:,1); x2 = outer(:,1); x3 = inner(:,1);                         % Example data for each class
n1 = length(x1); n2 = length(x2); n3 = length(x3);                          
xr1 = AR_filter(x1,700); xr2 = AR_filter(x2,700); xr3 = AR_filter(x3,700);  % AR filtering
xb1 = skbp(xr1,fs); xb2 = skbp(xr2,fs); xb3 = skbp(xr3,fs);                 % SK, bandpass filtering
xn1 = abs(hilbert(xb1)); xn2 = abs(hilbert(xb2)); xn3 = abs(hilbert(xb3));  % Envelope
xn1 = xn1-mean(xn1); xn2 = xn2-mean(xn2); xn3 = xn3-mean(xn3);
X1 = abs(fft(xn1))/n1*2; X2 = abs(fft(xn2))/n2*2; X3 = abs(fft(xn3))/n3*2;  % FFT
X1 = X1(1:ceil(n1/2)); X2 = X2(1:ceil(n2/2)); X3 = X3(1:ceil(n3/2));
f1 = (0:n1-1)'/n1*fs; f2 = (0:n2-1)'/n2*fs; f3 = (0:n3-1)'/n3*fs;                 
f1 = f1(1:ceil(n1/2)); f2 = f2(1:ceil(n2/2)); f3 = f3(1:ceil(n3/2));
%%
% Normal bearing <Figure 11(a)>
figure(1); stem(f1,X1,'linewidth',1.5); xlim([0 200]);
xline(bff(1)*fr,'r--','linewidth',2); xline(bff(2)*fr,'g--','linewidth',2);
xline(bff(3)*fr,'b--','linewidth',2); xline(bff(4)*fr,'k--','linewidth',2);
title('Normal'); xlabel('Frequency (Hz)'); ylim([0 0.0025]);
legend('Data','BPFO','BPFI','FTF','BSF','Location','Best');

% Outer race fault bearing <Figure 11(b)>
figure(2); stem(f2,X2,'linewidth',1.5); xlim([0 200]);
xline(bff(1)*fr,'r--','linewidth',2); xline(bff(2)*fr,'g--','linewidth',2);
xline(bff(3)*fr,'b--','linewidth',2); xline(bff(4)*fr,'k--','linewidth',2);
title('Outer race fault'); xlabel('Frequency (Hz)'); ylim([0 0.0025]);
legend('Data','BPFO','BPFI','FTF','BSF','Location','Best');

% Inner race fault bearing <Figure 11(c)>
figure(3); stem(f3,X3,'linewidth',1.5); xlim([0 200]);
xline(bff(1)*fr,'r--','linewidth',2); xline(bff(2)*fr,'g--','linewidth',2);
xline(bff(3)*fr,'b--','linewidth',2); xline(bff(4)*fr,'k--','linewidth',2);
title('Inner race fault'); xlabel('Frequency (Hz)'); ylim([0 0.0025]);
legend('Data','BPFO','BPFI','FTF','BSF','Location','Best');

% 2. Feature extraction
features_normal = []; features_outer = []; features_inner = [];
% Normal
for ix = 1:10
    x = normal(:,ix);
    [tmp1,fn_time] = TimeFeatures(x);
    xr = AR_filter(x,700);
    [tmp2,fn_bearing] = Bear_feat(xr,fs,bff*fr,cutoff);
    features_normal = [features_normal; tmp1 tmp2];
end
% Outer
for ix = 1:10
    x = outer(:,ix);
    [tmp1,~] = TimeFeatures(x);
    xr = AR_filter(x,700);
    [tmp2,~] = Bear_feat(xr,fs,bff*fr,cutoff);
    features_outer = [features_outer; tmp1 tmp2];
end
% Inner
for ix = 1:10
    x = inner(:,ix);
    [tmp1,~] = TimeFeatures(x);
    xr = AR_filter(x,700);
    [tmp2,~] = Bear_feat(xr,fs,bff*fr,cutoff);
    features_inner = [features_inner; tmp1 tmp2];
end
feature_name = [fn_time fn_bearing];

% Normalizing
features = [features_normal; features_outer; features_inner];
m = mean(features); s = std(features);
features = (features-m)./s;
ix_normal = 1:10; ix_outer = 11:20; ix_inner = 21:30;

% Features of CWRU bearing <Figrure 12>
figure; hold on;
for ix = 1:length(feature_name)
    plot(ix*ones(10,1),features(ix_normal,ix),'bo','markersize',7,'markerfacecolor','b');
    plot(ix*ones(10,1),features(ix_outer,ix),'rx','markersize',10);
    plot(ix*ones(10,1),features(ix_inner,ix),'gd','markersize',7,'markerfacecolor','g');
end
xticks([1:length(feature_name)]); xticklabels(feature_name);
xlim([0 length(feature_name)+1]); ylabel('Normalized value');
legend('Normal','Outer','Inner','Location','Best');
title('Feature of CWRU bearing');
set(gcf,'Position',[-1527 185 1351 483]);


% 3. Scatter matrix (J3)
tmp = nchoosek(1:length(feature_name),2); 
lab = [zeros(10,1); ones(10,1); 2*ones(10,1)];
for ix = 1:length(tmp)
    J3(ix) = ScattMat([features(:,tmp(ix,1)),features(:,tmp(ix,2))],lab);
end
[J3,ix] = sort(J3,'descend');

% Best3 <Table 12>
Top3 = [1;2;3];
Feature_combination = [{[feature_name{tmp(ix(1),1)},' & ',feature_name{tmp(ix(1),2)}]};...
    {[feature_name{tmp(ix(2),1)},' & ',feature_name{tmp(ix(2),2)}]};...
    {[feature_name{tmp(ix(3),1)},' & ',feature_name{tmp(ix(3),2)}]}];
J3_value = [J3(1); J3(2); J3(3)];
T1 = table(Top3,Feature_combination,J3_value)

% Worst3 <Table 12>
Bottom3 = [1;2;3];
Feature_combination = [{[feature_name{tmp(ix(end),1)},' & ',feature_name{tmp(ix(end),2)}]};...
    {[feature_name{tmp(ix(end-1),1)},' & ',feature_name{tmp(ix(end-1),2)}]};...
    {[feature_name{tmp(ix(end-2),1)},' & ',feature_name{tmp(ix(end-2),2)}]}];
J3_value = [J3(end); J3(end-1); J3(end-2)];
T2 = table(Bottom3,Feature_combination,J3_value)

% High J3 value <Figure 14(a)>
figure; hold on;
plot(features(ix_normal,tmp(ix(1),1)),features(ix_normal,tmp(ix(1),2)),'bo','markersize',7,'markerfacecolor','b');
plot(features(ix_outer,tmp(ix(1),1)),features(ix_outer,tmp(ix(1),2)),'rx','markersize',10);
plot(features(ix_inner,tmp(ix(1),1)),features(ix_inner,tmp(ix(1),2)),'gd','markersize',7,'markerfacecolor','g');
xlabel(feature_name{tmp(ix(1),1)}); ylabel(feature_name{tmp(ix(1),2)});
title(['Scatter plot (J3 = ',num2str(J3(1)),')']);
legend('Normal','Outer','Inner','Location','Best');

% Low J3 value <Figure 14(b)>
figure; hold on;
plot(features(ix_normal,tmp(ix(end),1)),features(ix_normal,tmp(ix(end),2)),'bo','markersize',7,'markerfacecolor','b');
plot(features(ix_outer,tmp(ix(end),1)),features(ix_outer,tmp(ix(end),2)),'rx','markersize',10);
plot(features(ix_inner,tmp(ix(end),1)),features(ix_inner,tmp(ix(end),2)),'gd','markersize',7,'markerfacecolor','g');
xlabel(feature_name{tmp(ix(end),1)}); ylabel(feature_name{tmp(ix(end),2)});
title(['Scatter plot (J3 = ',num2str(J3(end)),')']);
legend('Normal','Outer','Inner','Location','Best');