close all
clearvars
clc

fprintf(datestr(datetime(now,'ConvertFrom','datenum')))
fprintf('\n')

% Transmit power of the source in dBm, e.g. 200mW = 23dBm
Power_Source = -40:5:80;
% Noise power and Transmit power P_S
% Bandwidth
BW = 10e6; % 10 MHz
% Noise figure (in dB)
noiseFiguredB = 10;
% Compute the noise power in dBm
sigma2dBm = -174 + 10*log10(BW) + noiseFiguredB; % -94 dBm

N = 4; % Number of RIS elements
kappa_nl = 1; % Amplitude reflection coefficient
Q = 8; % Number of phase-shift levels

antenna_gain_S = db2pow(5); % Source antenna gain, dBi
antenna_gain_RIS = db2pow(5); % Gain of each element of a RIS, dBi
antenna_gain_D = db2pow(0); % Destination antenna gain, dBi
antenna_gain_E = db2pow(0); % Destination antenna gain, dBi

position = 1000;
vmean = 2; % D speed average
tmean = pi; % D direction average
vsigma = 0.1; % standard deviation of D speed
tsigma = pi/2; % standard deviation of D direction

% group member
Vmax = 3;
Vrange = 0.01;
Trange = pi/10;
alpha = 0.6; % memory level: 1=markov, 0=gauss

% simulation area
Xmax = 100;
Ymax = 100;

speed_g = zeros([position,1]);
theta_g = zeros([position,1]);
speed_g(1) = rand+vmean;
theta_g(1) = rand*2*pi;

speed_eve = zeros([position,1]);
theta_eve = zeros([position,1]);

Xdes = zeros([position,1]);
Ydes = zeros([position,1]);
Zdes = zeros([position,1]);
Xuav = zeros([position,1]);
Yuav = zeros([position,1]);
Zuav = zeros([position,1]);
Xsrc = zeros([position,1]);
Ysrc = zeros([position,1]);
Zsrc = zeros([position,1]);
Xeve = zeros([position,1]);
Yeve = zeros([position,1]);
Zeve = zeros([position,1]);

% Random first position of every node
Xnorm = normrnd(Xmax/2,Xmax/10,[4,1]);
Ynorm = normrnd(Ymax/2,Ymax/10,[4,1]);

Xsrc(1) = Xnorm(1);
Ysrc(1) = Ynorm(1);
Zsrc(1) = 3;

Xuav(1) = Xnorm(2);
Yuav(1) = Ynorm(2);
Zuav(1) = 10;

Xdes(1) = Xnorm(3);
Ydes(1) = Ynorm(3);
Zdes(1) = 4;

Xeve(1) = Xnorm(4);
Yeve(1) = Ynorm(4);
Zeve(1) = 2;

for t = 2:position
    speed_g(t)=(alpha*speed_g(t-1))+((1-alpha)*vmean)+...
        (vsigma*sqrt((1-(alpha^2))*normrnd(vmean,vsigma)));
    theta_g(t)=wrapTo2Pi(abs((alpha*theta_g(t-1))+((1-alpha)*tmean)+...
        (tsigma*sqrt((1-(alpha^2))*normrnd(tmean,tsigma)))));
    [Xdes(t),Ydes(t),theta_g(t)] = func_next_position...
        (Xdes(t-1),Ydes(t-1),speed_g(t),theta_g(t),Xmax,Ymax);
    Zdes(t) = Zdes(t-1);

    speed_uav = min(max(speed_g(t)+rand*Vrange,0),Vmax);
    theta_uav = wrapTo2Pi(theta_g(t) + rand*Trange);
    [Xuav(t),Yuav(t),theta_uav] = func_next_position...
        (Xuav(t-1),Yuav(t-1),speed_uav,theta_uav,Xmax,Ymax);
    Zuav(t) = Zuav(t-1);
    
    speed_src = min(max(speed_g(t)+rand*Vrange,0),Vmax);
    theta_src = wrapTo2Pi(theta_g(t) + rand*Trange);
    [Xsrc(t),Ysrc(t),theta_src] = func_next_position...
        (Xsrc(t-1),Ysrc(t-1),speed_src,theta_src,Xmax,Ymax);
    Zsrc(t) = Zsrc(t-1);
    
    % random walk
    speed_eve(t) = rand*2*Vmax;
    theta_eve(t) = rand*2*pi;
    [Xeve(t),Yeve(t),theta_eve(t)] = func_next_position...
        (Xeve(t-1),Yeve(t-1),speed_eve(t),theta_eve(t),Xmax,Ymax);
    Zeve(t) = Zeve(t-1);
end

pos_src(:,1) = Xsrc;
pos_src(:,2) = Ysrc;
pos_src(:,3) = Zsrc;
pos_uav(:,1) = Xuav;
pos_uav(:,2) = Yuav;
pos_uav(:,3) = Zuav;
pos_des(:,1) = Xdes;
pos_des(:,2) = Ydes;
pos_des(:,3) = Zdes;
pos_eve(:,1) = Xeve;
pos_eve(:,2) = Yeve;
pos_eve(:,3) = Zeve;

dSU = sqrt(sum((pos_src - pos_uav).^2 , 2));
dUD = sqrt(sum((pos_uav - pos_des).^2 , 2));
dUE = sqrt(sum((pos_uav - pos_eve).^2 , 2));

fc = 3; % GHz
% 3GPP Urban Micro in 3GPP TS 36.814, Mar. 2010.
% Note that x is measured in meter
% NLoS path-loss component based on distance
pathloss_NLOS = @(x) db2pow(-22.7 - 26*log10(fc) - 36.7*log10(x));

% phase of channels  <-- phase is random
phase_h_SU = 2*pi*rand(N, position); % domain [0,2pi)
phase_g_UD = 2*pi*rand(N, position); % domain [0,2pi)
phase_g_UE = 2*pi*rand(N, position); % domain [0,2pi)

h_SU = zeros([N,position]);
g_UD = zeros([N,position]);
g_UE = zeros([N,position]);
for pp = 1:position
    path_loss_SU = pathloss_NLOS(dSU(pp)) * ...
        antenna_gain_S*antenna_gain_RIS*N; % Source -> RIS
    path_loss_UD = pathloss_NLOS(dUD(pp)) * ...
        antenna_gain_RIS*N*antenna_gain_D; % RIS -> Des
    path_loss_UE = pathloss_NLOS(dUE(pp)) * ...
        antenna_gain_RIS*N*antenna_gain_E; % RIS -> Eve
    h_SU(:,pp) = sqrt(path_loss_SU) .* ...
        random('Rayleigh', sqrt(1/2), [N, 1]) .* ...
        exp(1i*phase_h_SU(:,pp));
    g_UD(:,pp) = sqrt(path_loss_UD) .* ...
        random('Rayleigh', sqrt(1/2), [N, 1]) .* ...
        exp(1i*phase_g_UD(:,pp));
    g_UE(:,pp) = sqrt(path_loss_UE) .* ...
        random('Rayleigh', sqrt(1/2), [N, 1]) .* ...
        exp(1i*phase_g_UE(:,pp));
end

% For RPS
tic
ps_RDP = 2*pi*randi([1 Q],N,position)/Q;
ps_el_RDP = wrapTo2Pi(ps_RDP);
ps_vec_RDP = exp(1i*ps_el_RDP);
h_SUD_RDP = zeros([length(Power_Source),position]);
h_SUE_RDP = zeros([length(Power_Source),position]);
SC_RDP = zeros([length(Power_Source),1]);
SC_RDP_pos_all = zeros([length(Power_Source),position]);
pe_RDP = zeros([N,position]);
for pow = 1:length(Power_Source)
    avgSNR = db2pow(Power_Source(pow) - sigma2dBm);
    for pp = 1:position
        pe_RDP(:,pp)=wrapTo2Pi(abs(ps_el_RDP(:,pp)-phase_h_SU(:,pp)-phase_g_UD(:,pp)));
        ps_matrix_RDP = kappa_nl .* diag(ps_vec_RDP(:,pp));
        h_SUD_RDP(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_RDP*g_UD(:,pp)).^2;
        h_SUE_RDP(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_RDP*g_UE(:,pp)).^2;
        rateD_RDP = log2(1 + avgSNR * h_SUD_RDP(pow,pp));
        rateE_RDP = log2(1 + avgSNR * h_SUE_RDP(pow,pp));
        SC_RDP_pos_all(pow,pp) = max(rateD_RDP - rateE_RDP, 0);
    end
    SC_RDP(pow) = mean(SC_RDP_pos_all(pow,:));
end
ps_RDP_deg = rad2deg(ps_el_RDP);
timeRDP = toc;

ps_conf = zeros([Q,1]);
for qq = 1:Q
    ps_conf(qq) = wrapTo2Pi(2*pi*qq/Q);
end

% For MRC
tic
ps_el_MMRC = zeros([N,position]);
ps_idx_MMRC = zeros([N,position]);
ps_vec_MMRC = zeros([length(Power_Source),N]);
h_SUD_check = zeros([N,Q,position]);
h_SUD_MMRC = zeros([length(Power_Source),position]);
h_SUE_MMRC = zeros([length(Power_Source),position]);
SC_MMRC = zeros([length(Power_Source),1]);
SC_MMRC_pos_all = zeros([length(Power_Source),position]);
for pow = 1:length(Power_Source)
    avgSNR = db2pow(Power_Source(pow) - sigma2dBm);
    for pp = 1:position
        for nn = 1:N
            for qq = 1:Q
                h_SUD_check(nn,qq,pp) = h_SU(nn,pp)*kappa_nl*...
                    exp(1i*ps_conf(qq))*g_UD(nn,pp);
            end
            [max_val,ps_idx_MMRC(nn,pp)]=max(real(h_SUD_check(nn,:,pp)));
            ps_el_MMRC(nn,pp) = ps_conf(ps_idx_MMRC(nn,pp));
        end
        ps_vec_MMRC(pow,:) = exp(1i*ps_el_MMRC(:,pp));
        ps_matrix_MMRC = kappa_nl .* diag(ps_vec_MMRC(pow,:));
        h_SUD_MMRC(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_MMRC*g_UD(:,pp)).^2;
        h_SUE_MMRC(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_MMRC*g_UE(:,pp)).^2;
        rateD_MMRC = log2(1 + avgSNR * h_SUD_MMRC(pow,pp));
        rateE_MMRC = log2(1 + avgSNR * h_SUE_MMRC(pow,pp));
        SC_MMRC_pos_all(pow,pp) = max(rateD_MMRC - rateE_MMRC, 0);
    end
    SC_MMRC(pow) = mean(SC_MMRC_pos_all(pow,:));
end
ps_MMRC_deg = rad2deg(ps_el_MMRC);
timeMMRC = toc;

% For OSP
tic
SC_OSP = zeros([length(Power_Source),1]);
SC_OSP_pos_all = zeros([length(Power_Source),position]);
OSP_index = zeros([length(Power_Source),position]);
ps_OSP = zeros([N,position,length(Power_Source)]);
h_SUD_OSP = zeros([length(Power_Source),position]);
h_SUE_OSP = zeros([length(Power_Source),position]);
h_main_OSP = zeros([length(Power_Source),Q^N,position]);
h_main_max = zeros([length(Power_Source),position]);
h_main_idx = zeros([length(Power_Source),position]);
for pow = 1:length(Power_Source)
    avgSNR = db2pow(Power_Source(pow) - sigma2dBm);
    for pp = 1:position
        for idx = 1:Q^N
            ps_el_OSP = zeros([N,1]);
            for el = 1:N-1
                ps_el_OSP(el) = wrapTo2Pi(2*pi*...
                    ceil(idx/(Q^(N-el)))/Q);
            end
            ps_el_OSP(N) = wrapTo2Pi(2*pi*idx/Q);
            ps_vec_OSP = exp(1i*ps_el_OSP);
            ps_matrix_OSP = kappa_nl .* diag(ps_vec_OSP);
            h_main_OSP(pow,idx,pp) = abs(h_SU(:,pp).'*ps_matrix_OSP*g_UD(:,pp)).^2;
            h_eve_OSP = abs(h_SU(:,pp).'*ps_matrix_OSP*g_UE(:,pp)).^2;
            rateD_OSP = log2(1 + avgSNR * h_main_OSP(pow,idx,pp));
            rateE_OSP = log2(1 + avgSNR * h_eve_OSP);
            SC_OSP_data = max(rateD_OSP-rateE_OSP, 0);
            if SC_OSP_data > SC_OSP_pos_all(pow,pp)
                SC_OSP_pos_all (pow,pp) = SC_OSP_data;
                OSP_index (pow,pp) = idx;
            end
        end
        [h_main_max(pow,pp),h_main_idx(pow,pp)] = max(h_main_OSP(pow,:,pp));
        for el = 1:N-1
            ps_OSP(el,pp,pow) = wrapTo2Pi(2*pi*...
                ceil(OSP_index(pow,pp)/(Q^(N-el)))/Q);
        end
        ps_OSP(N,pp,pow)=wrapTo2Pi(2*pi*OSP_index(pow,pp)/Q);
        ps_vec_OSP = exp(1i*ps_OSP(:,pp,pow));
        ps_matrix_OSP = kappa_nl .* diag(ps_vec_OSP);
        h_SUD_OSP(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_OSP*g_UD(:,pp)).^2;
        h_SUE_OSP(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_OSP*g_UE(:,pp)).^2;
    end
    SC_OSP(pow) = mean(SC_OSP_pos_all(pow,:));
end
ps_OSP_deg = rad2deg(ps_OSP);
timeOSP = toc;

nomatch = (h_SUD_MMRC == h_main_max);
quantity = sum(nomatch,2);

Q = 16;

% For RPS2
tic
ps_RDP2 = 2*pi*randi([1 Q],N,position)/Q;
ps_el_RDP2 = wrapTo2Pi(ps_RDP2);
ps_vec_RDP2 = exp(1i*ps_el_RDP2);
h_SUD_RDP2 = zeros([length(Power_Source),position]);
h_SUE_RDP2 = zeros([length(Power_Source),position]);
SC_RDP2 = zeros([length(Power_Source),1]);
SC_RDP_pos_all2 = zeros([length(Power_Source),position]);
pe_RDP2 = zeros([N,position]);
for pow = 1:length(Power_Source)
    avgSNR = db2pow(Power_Source(pow) - sigma2dBm);
    for pp = 1:position
        pe_RDP2(:,pp)=wrapTo2Pi(abs(ps_el_RDP2(:,pp)-phase_h_SU(:,pp)-phase_g_UD(:,pp)));
        ps_matrix_RDP2 = kappa_nl .* diag(ps_vec_RDP2(:,pp));
        h_SUD_RDP2(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_RDP2*g_UD(:,pp)).^2;
        h_SUE_RDP2(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_RDP2*g_UE(:,pp)).^2;
        rateD_RDP2 = log2(1 + avgSNR * h_SUD_RDP2(pow,pp));
        rateE_RDP2 = log2(1 + avgSNR * h_SUE_RDP2(pow,pp));
        SC_RDP_pos_all2(pow,pp) = max(rateD_RDP2 - rateE_RDP2, 0);
    end
    SC_RDP2(pow) = mean(SC_RDP_pos_all2(pow,:));
end
ps_RDP2_deg = rad2deg(ps_el_RDP2);
timeRDP2 = toc;

ps_conf2 = zeros([Q,1]);
for qq = 1:Q
    ps_conf2(qq) = wrapTo2Pi(2*pi*qq/Q);
end

% For MRC2
tic
ps_el_MMRC2 = zeros([N,position]);
ps_idx_MMRC2 = zeros([N,position]);
ps_vec_MMRC2 = zeros([length(Power_Source),N]);
h_SUD_check2 = zeros([N,Q,position]);
h_SUD_MMRC2 = zeros([length(Power_Source),position]);
h_SUE_MMRC2 = zeros([length(Power_Source),position]);
SC_MMRC2 = zeros([length(Power_Source),1]);
SC_MMRC_pos_all2 = zeros([length(Power_Source),position]);
for pow = 1:length(Power_Source)
    avgSNR = db2pow(Power_Source(pow) - sigma2dBm);
    for pp = 1:position
        for nn = 1:N
            for qq = 1:Q
                h_SUD_check2(nn,qq,pp) = h_SU(nn,pp)*kappa_nl*...
                    exp(1i*ps_conf2(qq))*g_UD(nn,pp);
            end
            [max_val2,ps_idx_MMRC2(nn,pp)]=max(real(h_SUD_check2(nn,:,pp)));
            ps_el_MMRC2(nn,pp) = ps_conf2(ps_idx_MMRC2(nn,pp));
        end
        ps_vec_MMRC2(pow,:) = exp(1i*ps_el_MMRC2(:,pp));
        ps_matrix_MMRC2 = kappa_nl .* diag(ps_vec_MMRC2(pow,:));
        h_SUD_MMRC2(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_MMRC2*g_UD(:,pp)).^2;
        h_SUE_MMRC2(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_MMRC2*g_UE(:,pp)).^2;
        rateD_MMRC2 = log2(1 + avgSNR * h_SUD_MMRC2(pow,pp));
        rateE_MMRC2 = log2(1 + avgSNR * h_SUE_MMRC2(pow,pp));
        SC_MMRC_pos_all2(pow,pp) = max(rateD_MMRC2 - rateE_MMRC2, 0);
    end
    SC_MMRC2(pow) = mean(SC_MMRC_pos_all2(pow,:));
end
ps_MMRC2_deg = rad2deg(ps_el_MMRC2);
timeMMRC2 = toc;

% For OSP2
tic
SC_OSP2 = zeros([length(Power_Source),1]);
SC_OSP_pos_all2 = zeros([length(Power_Source),position]);
OSP_index2 = zeros([length(Power_Source),position]);
ps_OSP2 = zeros([N,position,length(Power_Source)]);
h_SUD_OSP2 = zeros([length(Power_Source),position]);
h_SUE_OSP2 = zeros([length(Power_Source),position]);
h_main_OSP2 = zeros([length(Power_Source),Q^N,position]);
h_main_max2 = zeros([length(Power_Source),position]);
h_main_idx2 = zeros([length(Power_Source),position]);
for pow = 1:length(Power_Source)
    avgSNR = db2pow(Power_Source(pow) - sigma2dBm);
    for pp = 1:position
        for idx = 1:Q^N
            ps_el_OSP2 = zeros([N,1]);
            for el = 1:N-1
                ps_el_OSP2(el) = wrapTo2Pi(2*pi*...
                    ceil(idx/(Q^(N-el)))/Q);
            end
            ps_el_OSP2(N) = wrapTo2Pi(2*pi*idx/Q);
            ps_vec_OSP2 = exp(1i*ps_el_OSP2);
            ps_matrix_OSP2 = kappa_nl .* diag(ps_vec_OSP2);
            h_main_OSP2(pow,idx,pp)=abs(h_SU(:,pp).'*ps_matrix_OSP2*g_UD(:,pp)).^2;
            h_eve_OSP2 = abs(h_SU(:,pp).'*ps_matrix_OSP2*g_UE(:,pp)).^2;
            rateD_OSP2 = log2(1 + avgSNR * h_main_OSP2(pow,idx,pp));
            rateE_OSP2 = log2(1 + avgSNR * h_eve_OSP2);
            SC_OSP_data2 = max(rateD_OSP2-rateE_OSP2, 0);
            if SC_OSP_data2 > SC_OSP_pos_all2(pow,pp)
                SC_OSP_pos_all2 (pow,pp) = SC_OSP_data2;
                OSP_index2 (pow,pp) = idx;
            end
        end
        [h_main_max2(pow,pp),h_main_idx2(pow,pp)] = max(h_main_OSP2(pow,:,pp));
        for el = 1:N-1
            ps_OSP2(el,pp,pow) = wrapTo2Pi(2*pi*...
                ceil(OSP_index(pow,pp)/(Q^(N-el)))/Q);
        end
        ps_OSP2(N,pp,pow)=wrapTo2Pi(2*pi*OSP_index(pow,pp)/Q);
        ps_vec_OSP2 = exp(1i*ps_OSP2(:,pp,pow));
        ps_matrix_OSP2 = kappa_nl .* diag(ps_vec_OSP2);
        h_SUD_OSP2(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_OSP2*g_UD(:,pp)).^2;
        h_SUE_OSP2(pow,pp)=abs(h_SU(:,pp).'*ps_matrix_OSP2*g_UE(:,pp)).^2;
    end
    SC_OSP2(pow) = mean(SC_OSP_pos_all2(pow,:));
end
ps_OSP2_deg = rad2deg(ps_OSP2);
timeOSP2 = toc;

figure(1)
plot3(Xsrc,Ysrc,Zsrc,'gs-','LineWidth',1.5)
hold on
plot3(Xuav,Yuav,Zuav,'k>-','LineWidth',1.5)
plot3(Xdes,Ydes,Zdes,'bo-','LineWidth',1.5)
plot3(Xeve,Yeve,Zeve,'rd-','LineWidth',1.5)
hold off; grid on
xlabel('$x$', 'interpreter', 'latex')
ylabel('$y$', 'interpreter', 'latex')
zlabel('$z$', 'interpreter', 'latex')
legend('Source','RIS-UAV','Destination','Eavesdropper',...
    'location','best','interpreter','latex')

figure(2)
plot(Power_Source,SC_OSP,'r*-','linewidth',1.2)
hold on
plot(Power_Source,SC_MMRC,'b^-','linewidth',1.2)
plot(Power_Source,SC_RDP,'ko-','linewidth',1.2)
plot(Power_Source,SC_OSP2,'r*:','linewidth',1.2)
plot(Power_Source,SC_MMRC2,'b^:','linewidth',1.2)
plot(Power_Source,SC_RDP2,'ko:','linewidth',1.2)
hold off
legend('OSPC(Q=8)','MMRC(Q=8)','RPSC(Q=8)','OSPC(Q=16)',...
    'MMRC(Q=16)','RPSC(Q=16)','Location','best')
xlabel('Transmit SNR (dB)')
ylabel('Average Secrecy Rate (bps/Hz)')

% figure(3)
% plot(SC_OSP_pos_all(15,:),'r*-','linewidth',1.2)
% hold on
% plot(SC_MMRC_pos_all(15,:),'b^-','linewidth',1.2)
% plot(SC_RDP_pos_all(15,:),'ko-','linewidth',1.2)
% plot(SC_OSP_pos_all2(15,:),'r*:','linewidth',1.2)
% plot(SC_MMRC_pos_all2(15,:),'b^:','linewidth',1.2)
% plot(SC_RDP_pos_all2(15,:),'ko:','linewidth',1.2)
% hold off
% legend('OSPC(Q=8)','MMRC(Q=8)','RPSC(Q=8)','OSPC(Q=16)',...
%     'MMRC(Q=16)','RPSC(Q=16)','Location','best')
% xlabel('Location Index')
% ylabel('Instantaneous Secrecy Capacity (bps/Hz)')

values = zeros([length(Power_Source)*position,1+12+3+(3*N)+(2*3*N)+(5*6)]);

rr = 1;
for pow = 1 : length(Power_Source)
    for pp = 1 : position
        values(rr,1:(1+12+3))=[Power_Source(pow),Xsrc(pp),Ysrc(pp),...
            Xuav(pp),Yuav(pp),Xdes(pp),Ydes(pp),Xeve(pp),Yeve(pp),...
            Zsrc(pp),Zuav(pp),Zdes(pp),Zeve(pp),dSU(pp),dUD(pp),dUE(pp)];
        for nn = 1 : N
            values(rr, 16+nn) = phase_h_SU(nn,pp);
            values(rr, 16+N+nn) = phase_g_UD(nn,pp);
            values(rr, 16+(2*N)+nn) = phase_g_UE(nn,pp);
            values(rr, 28+((nn-1)*2)+1) = real(h_SU(nn,pp));
            values(rr, 28+((nn-1)*2)+2) = imag(h_SU(nn,pp));
            values(rr, 36+((nn-1)*2)+1) = real(g_UD(nn,pp));
            values(rr, 36+((nn-1)*2)+2) = imag(g_UD(nn,pp));
            values(rr, 44+((nn-1)*2)+1) = real(g_UE(nn,pp));
            values(rr, 44+((nn-1)*2)+2) = imag(g_UE(nn,pp));
            values(rr, 52+nn) = ps_el_RDP(nn,pp);
            values(rr, 56+nn) = ps_el_RDP2(nn,pp);
            values(rr, 60+nn) = ps_el_MMRC(nn,pp);
            values(rr, 64+nn) = ps_el_MMRC2(nn,pp);
            values(rr, 68+nn) = ps_OSP(nn,pp,pow);
            values(rr, 72+nn) = ps_OSP2(nn,pp,pow);
        end
        values(rr,77:82)=[SC_RDP_pos_all(pow,pp),SC_RDP_pos_all2(pow,pp),...
            SC_MMRC_pos_all(pow,pp),SC_MMRC_pos_all2(pow,pp),...
            SC_OSP_pos_all(pow,pp),SC_OSP_pos_all2(pow,pp)];
        rr = rr + 1;
    end
end
