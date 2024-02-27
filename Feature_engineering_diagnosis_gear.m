set(0,'defaultAxesFontSize',15); set(0,'defaultLineLinewidth',1.5);        
set(0,'DefaultTextInterpreter','none');   
%% HS Gear
% ============================ Explanation ================================
% This code is intended to show the signal processing in the paper
% step-by-step using HS gear
% This dataset was originally downloaded at data-acoustics.com, 
% which was developed by Dr. Eric Bechhoefer, but unavailable now.
% Dr. Bechhoefer gave us permission to distribute the dataset at our site, 
% which is greatly appreciated.
% https://www.kau-sdol.com/kaug
% 1. Signal processing (TSA)
% 2. Feature extraction (Time features & features for gear)
% 3. Feature selection (FDR)
% Two class gear data with normal and fault
% =========================================================================
clear; clc; close all;
currentFolder = pwd;
addpath(currentFolder);
addpath('Functions');                                       % Input function folder directory
cd('Data_repository/HS_gear');                              % Input data folder directory
fr = 30;                                                    % Rotating speed
Nr = 1;                                                     % Number of rotations for TSA
% ===================== 1. Load normal data & TSA =========================
% Normal
cd('data2');
file = dir('*.mat'); N_normal = length(file);
ix_example = 2;
for ix_file = 1:N_normal
    load(file(ix_file).name);
    [normal{ix_file},t] = sdol_tsa(gs,sr,tach,ppr,Nr);         % TSA
    % Example to ililustrate effects of TSA
    if ix_file == ix_example
        x = gs;                 % Raw data of example
        t2 = t;                 % Time vector for TSA signal of example
    end
end
cd('..');

% Fault
cd('data1');
file = dir('*.mat'); N_fault = length(file);
for ix_file = 1:N_fault
    load(file(ix_file).name);
    [fault{ix_file},~] = sdol_tsa(gs,sr,tach,ppr,Nr);          % TSA
end
cd('..');

% Time domain signal <Figure 3(a)>
t = (0:1/sr:t2(end))'; N = length(t); 
x1 = x(1:N); x2 = normal{ix_example}; 
figure(1);
subplot(211); plot(t,x1); title('Raw data');
subplot(212); plot(t2,x2); title('TSA data'); xlabel('Time(s)');

% Frequency domain signal <Figure 3(b)>
f = (0:N-1)'/N*sr; f = f(1:ceil(N/2));
X1 = abs(fft(x1))/N*2; X1 = X1(1:ceil(N/2));
X2 = abs(fft(x2))/N*2; X2 = X2(1:ceil(N/2));
figure(2);
subplot(211); stem(f,X1); title('Raw data'); xlim([400 1400]);
subplot(212); stem(f,X2); title('TSA data'); xlim([400 1400]); 
xlabel('Frequency(Hz)');

% ======================== 2. Feature extraction ==========================
features_normal = []; features_fault = [];
% Normal
for ix = 1:N_normal
    [tmp1,fn_time] = TimeFeatures(normal{ix});
    [tmp2,fn_gear] = Gear_feat(normal{ix},teeth,sr,fr);
    features_normal = [features_normal; tmp1 tmp2];
end

% Fault
for ix = 1:N_fault
    [tmp1,~] = TimeFeatures(fault{ix});
    [tmp2,~] = Gear_feat(fault{ix},teeth,sr,fr);
    features_fault = [features_fault; tmp1 tmp2];
end

% Normalizing
features = [features_normal; features_fault];
m = mean(features); s = std(features);
features = (features-m)./s;
ix_normal = 1:N_normal; ix_fault = N_normal+1:N_normal+N_fault;

% Time domain features <Figure 8>
figure(3); hold on;
for j = 1:length(fn_time)
    plot(j*ones(N_normal,1),features(ix_normal,j),'bo','MarkerSize',5,'Markerfacecolor','b');
    plot(j*ones(N_fault,1),features(ix_fault,j),'rx','MarkerSize',10);
end
xticks([1:length(fn_time)]); xticklabels(fn_time); xlim([0 length(fn_time)+1]);
title('Time domain features of HS gear'); ylabel('Normalized value'); grid on;
set(gcf,'Position',[-1597 499 1016 420]);

% Features for gear <Figure 9>
figure(4); hold on;
for j = length(fn_time)+1:length(fn_time)+length(fn_gear)
    ix = j-length(fn_time);
    plot(ix*ones(N_normal,1),features(ix_normal,j),'bo','MarkerSize',5,'Markerfacecolor','b');
    plot(ix*ones(N_fault,1),features(ix_fault,j),'rx','MarkerSize',10);
end
xticks([1:length(fn_gear)]); xticklabels(fn_gear); xlim([0 length(fn_gear)+1]);
title('Features for gear of HS gear'); ylabel('Normalized value'); grid on;
set(gcf,'Position',[-1597 499 1016 420]);

% ======================== 3. Features selection ==========================
feature_name = [fn_time fn_gear];
features_normal = features(ix_normal,:); features_fault = features(ix_fault,:);
m_normal = mean(features_normal); m_fault = mean(features_fault);
s_normal = std(features_normal); s_fault = std(features_fault);
fdr = (m_normal-m_fault).^2./(s_normal.^2+s_fault.^2);      % FDR
% Sorting
[fdr,ix] = sort(fdr,'descend');

% Best3 <Table 11>
Top3 = [1;2;3]; 
Feature = [feature_name(ix(1)); feature_name(ix(2)); feature_name(ix(3))];
FDR_value = [fdr(1); fdr(2); fdr(3)];
T1 = table(Top3,Feature,FDR_value)

% Worst3 <Table 11>
Bottom3 = [1;2;3]; 
Feature = [feature_name(ix(end)); feature_name(ix(end-1)); feature_name(ix(end-2))];
FDR_value = [fdr(end); fdr(end-1); fdr(end-2)];
T2 = table(Bottom3,Feature,FDR_value)

% High FDR value <Figure 13(a)>
x1 = m_normal(ix(1))-3*s_normal(ix(1)):0.01:m_normal(ix(1))+3*s_normal(ix(1));
pdf1 = normpdf(x1,m_normal(ix(1)),s_normal(ix(1)));
x2 = m_fault(ix(1))-3*s_fault(ix(1)):0.01:m_fault(ix(1))+3*s_fault(ix(1));
pdf2 = normpdf(x2,m_fault(ix(1)),s_fault(ix(1)));
figure(5); hold on;
plot(x1,pdf1,'b'); plot(x2,pdf2,'r--'); 
title([feature_name{ix(1)},' (FDR = ',num2str(fdr(1)),')']);
legend('Normal','Fault','Location','Best');

% Low FDR value <Figure 13(b)>
x1 = m_normal(ix(end))-3*s_normal(ix(end)):0.01:m_normal(ix(end))+3*s_normal(ix(end));
pdf1 = normpdf(x1,m_normal(ix(end)),s_normal(ix(end)));
x2 = m_fault(ix(end))-3*s_fault(ix(end)):0.01:m_fault(ix(end))+3*s_fault(ix(end));
pdf2 = normpdf(x2,m_fault(ix(end)),s_fault(ix(end)));
figure(6); hold on;
plot(x1,pdf1,'b'); plot(x2,pdf2,'r--'); 
title([feature_name{ix(end)},' (FDR = ',num2str(fdr(end)),')']);
legend('Normal','Fault','Location','Best');