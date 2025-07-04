external_noise = load('external_noise.txt');
noisy_speech = load('noisy_speech.txt');
clean_speech = load('clean_speech.txt');

fs = 44100; 
L = 4;
L_rls = 8;
mu = 0.005;
lam = 0.9999;
delta = 1e-3;
L_values = [4, 8, 16, 32, 64]; 

[vhat_lms] = lms(external_noise, noisy_speech, L, mu);
[vhat_nlms] = nlms(external_noise, noisy_speech, L, mu);
[vhat_rls] = rls(external_noise, noisy_speech, L_rls, lam, delta);

est_speech_lms = noisy_speech - vhat_lms;
est_speech_nlms = noisy_speech - vhat_nlms;
est_speech_rls = noisy_speech - vhat_rls;

noise_est_lms = clean_speech - est_speech_lms;
noise_est_nlms = clean_speech - est_speech_nlms;
noise_est_rls = clean_speech - est_speech_rls;
snr_lms = calculate_snr(clean_speech, noise_est_lms);
snr_nlms = calculate_snr(clean_speech, noise_est_nlms);
snr_rls = calculate_snr(clean_speech, noise_est_rls);

figure('Name', 'SNR Comparison', 'NumberTitle', 'off');

snr_values = [snr_lms, snr_nlms, snr_rls];

b = bar(1:3, snr_values);
set(gca, 'XTick', 1:3, 'XTickLabel', {'LMS', 'NLMS', 'RLS'});
ylabel('SNR (dB)');
title('SNR Comparison Across Adaptive Filters');
grid on;

fprintf('SNR Results:\n');
fprintf('LMS: %.3f dB\n', snr_lms);
fprintf('NLMS: %.3f dB\n', snr_nlms);
fprintf('RLS: %.3f dB\n', snr_rls);

figure;
snr_results_L = zeros(size(L_values));
lambda_const = 0.9999;
for i = 1:length(L_values)
    L = L_values(i);
    [vhat] = rls(external_noise, noisy_speech, L, lambda_const, delta);
    est_speech = noisy_speech - vhat;
    noise_est = clean_speech - est_speech;
    snr_results_L(i) = calculate_snr(clean_speech, noise_est);
end
plot(L_values, snr_results_L, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Filter Order (L)');
ylabel('SNR (dB)');
title('Variation of SNR with L (\lambda = 0.9999)');
grid on;

function snr = calculate_snr(signal, noise)
    signal_power = sum(signal.^2) / length(signal);
    noise_power = sum(noise.^2) / length(noise);
    snr = 10*log10(signal_power / noise_power);
end

function [yhat] = lms(x, y, L, mu)
    yhat = zeros(size(y));
    w = zeros(L,1);
    xn = zeros(L,1);
    for n = 1:length(y)
        xn = [x(n); xn(1:L-1)];
        yhat(n) = w'*xn;
        e = y(n)-yhat(n);
        w = w + mu*e*xn;
    end
end

function [yhat] = nlms(x, y, L, mu)
    yhat = zeros(size(y));
    w = zeros(L,1);
    xn = zeros(L,1);
    e = zeros(size(y));
    for n = 1:length(y)
        xn = [x(n); xn(1:L-1)];
        yhat(n) = w'*xn;
        e(n) = y(n)-yhat(n);
        step = mu/(1e-4+xn'*xn);
        w = w + step*xn*e(n);
    end
end

function [vhat] = rls(x, y, L, lam, delta)
    w = zeros(L,1);
    x_buff = zeros(L,1);
    vhat = zeros(size(y));
    P = eye(L)/delta;
    for i = 1:length(y)
        x_buff = [x(i); x_buff(1:L-1)];
        zn = P*x_buff;
        gn = zn/(lam+x_buff'*zn);
        vhat(i) = w'*x_buff;
        err = y(i)-vhat(i);
        w = w + gn*err;
        P = (P-gn*x_buff'*P)/lam;
    end
end
