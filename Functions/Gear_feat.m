function [feature, feature_name] = Gear_feat(tsa_sig,teeth,sr,fr)
% =========================================================================
% This code is programmed by System Design Optimization Lab (SDOL) at Korea
% Aerospace University (KAU)
% ============================= Input =====================================
% tsa_sig: TSA signal
% teeth: The number of teeth of gear
% sr: Sampling rate
% fr: Shaft rotational speed
% ============================= Output ====================================
% feature: Calculated feature value
% feature_name: The name of features
% =========================================================================

% FFT
N = length(tsa_sig); 
X = abs(fft(tsa_sig))/N*2; X = X(1:ceil(N/2));
f = (0:N-1)'/N*sr; f = f(1:ceil(N/2));
% Find rotating speed
ix = find(f>25 & f<35); fr = f(ix);
% Find GMF
gmf = teeth*fr; cutoff = sr/N*1.2;
ix = find(f>gmf-cutoff & f<gmf+cutoff);
gmf = f(find(X == max(X(ix))));         % Real GMF
cutoff = 10;
for hn = 1:10                           % Harmonic number
    P = gmf*hn;
    ix = find(f>P-cutoff & f<P+cutoff);
    gmf_amp(hn) = max(X(ix));
    for sn = 1:6                        % Sideband number
        S = fr*sn;
        ix_side1 = find(f>P-S-cutoff & f<P-S+cutoff);
        ix_side2 = find(f>P+S-cutoff & f<P+S+cutoff);
        side_amp(:,sn,hn) = [max(X(ix_side1)); max(X(ix_side2))];
    end
end

% ==================== Calculate Residual signal ==========================
res_sig = tsa_sig; ord = 2;
for hn = 1:10
    P = gmf*hn;
    [b,a] = butter(ord,[P-cutoff P+cutoff]/(sr/2),'stop');
    res_sig = filter(b,a,res_sig);
end
% =================== Calculate difference signal =========================
diff_sig = res_sig;
for hn = 1:10
    P = gmf*hn;
    [b,a] = butter(ord,[P-fr-cutoff P-fr+cutoff]/(sr/2),'stop');
    diff_sig = filter(b,a,diff_sig);
    [b,a] = butter(ord,[P+fr-cutoff P+fr+cutoff]/(sr/2),'stop');
    diff_sig = filter(b,a,diff_sig);
end
% =================== Calculate features for gear =========================
% 1. FM0
FM0 = (max(tsa_sig)-min(tsa_sig))/sum(gmf_amp);
% 2. SER
SER = sum(sum(side_amp(:,:,1)))/gmf_amp(1);
% 3. NA4
ress = res_sig - mean(res_sig);         
cur_ress = ress(:,end);                 				% Current signal
N = size(ress,1); M = size(ress,2);						% N x M : N samples M run ensemble
NA4 = N*sum(cur_ress.^4)/(sum((sum(ress.^2,1)),2)/M)^2;
% 4. FM4
diff = diff_sig - mean(diff_sig);
FM4 = length(diff)*sum(diff.^4)/(sum(diff.^2).^2);
% 5. M6A
M6A = length(diff)^2*sum(diff.^6)/(sum(diff.^2).^3);
% 6. M8A
M8A = length(diff)^2*sum(diff.^8)/(sum(diff.^2).^4);
% 7. ER
ER = rms(diff_sig)/sum(gmf_amp+squeeze(sum(sum(side_amp)))');
% 8. NB4
[a,b] = butter(ord,[gmf-fr gmf+fr]/(sr/2),'bandpass');
bp_sig = filter(a,b,tsa_sig);
env_bp_sig = abs(hilbert(bp_sig));                      % Envelope
s = env_bp_sig - mean(env_bp_sig);
cur_s = s(:,end);                       				% Current signal
N = size(s,1); M = size(s,2);           				% N x M : N samples M run ensemble
NB4 = N*sum(cur_s.^4)/(sum((sum(s.^2,1)),2)/M)^2;

feature = [FM0, SER, NA4, FM4, M6A, M8A, ER, NB4];
feature_name = {'FM0', 'SER', 'NA4', 'FM4', 'M6A', 'M8A', 'ER', 'NB4'};
end