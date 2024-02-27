set(0,'defaultAxesFontSize',15); set(0,'defaultLineLinewidth',1.5);        
set(0,'DefaultTextInterpreter','none');   
% ========================= Explanation ===================================
% This code describes the feature selection for prognosis using IMS bearing
% dataset
% Used data is IMS bearing dataset available at the link below
% https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
% Authors converted the data into mat file
% 1. Feature extraction (Time features, Features for bearing)
% 2. Feature selection for prognosis
% =========================================================================
%% IMS Bearing
clc; clear; close all;
currentFolder = pwd;
addpath(currentFolder);
addpath('Functions');                                      % Input function folder directory
cd('Data_repository\IMS_bearing');                          % Input data folder directory
file = dir('*.mat'); N_file = length(file);
% 1. Feature extraction
features = [];
fs = 20480; cutoff = 3; fr = 2000/60;
bff = [7.0921 8.9079 0.4433 4.1975];
for ix_file = 1:N_file
    load(file(ix_file).name);
    [tmp1,fn_time] = TimeFeatures(x);                       % Extract time domain features
    xr = AR_filter(x,700);
    xb = skbp(xr,fs);
    [tmp2,fn_bearing] = Bear_feat(xb,fs,bff*fr,cutoff);     % Extract features for bearing
    features = [features; tmp1 tmp2];
end
feature_name = [fn_time fn_bearing];

% 2. Feature selection for prognosis
cycle = (0:N_file-1)'; w = [0.33 0.33 0.33];

% Normalizing
m = mean(features); s = std(features);
features = (features-m)./s;

% Smoothing
for ix = 1:size(features,2)
    features_smth(:,ix) = smooth(features(:,ix),50);
end

% Monotonicity
mon = abs(sum(diff(features_smth)>0)-sum(diff(features_smth)<0))/(N_file-1);

% Trendability
tre = abs(corr(cycle,features_smth,'type','Spearman'));

% Robustness
rob = [];
for ix = 1:size(features,2)
    rob(ix) = mean(exp(-abs(features(:,ix)-features_smth(:,ix)./features(:,ix))));
end
cri = [mon' tre' rob']*w';
[cri_sort,ix] = sort(cri,'descend');

% Feature selection criteria <Figure 15(a)>
figure; bar(cri); title('Feature selection critera'); 
xticklabels(feature_name); set(gcf,'Position',[-1722 268 1359 631]);

% High criteria <Figure 15(b)>
figure; plot(cycle,features(:,ix(1)));
xlabel('Cycle'); ylabel('Normalized value');
title([feature_name{ix(1)},' (Criteria = ',num2str(cri_sort(1)),')']);

% Low criteria <Figure 15(c)>
figure; plot(cycle,features(:,ix(end)));
xlabel('Cycle'); ylabel('Normalized value');
title([feature_name{ix(end)},' (Criteria = ',num2str(cri_sort(end)),')']);