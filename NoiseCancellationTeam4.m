external_noise = load('external_noise.txt');
noisySpeech = load('noisy_speech.txt');
cleanSpeech = load('clean_speech.txt');

preserveTonalNoise = false;           
frequencies_to_suppress = [1000]; % Programmable Tonal Frequencies   
L = 8;                         
fs = 44100;                       
lambda = 0.9999;                  
delta = 0.001;                      

rls_w = zeros(L, 1);
rls_xn = zeros(L, 1);
rls_P = eye(L)/delta;
num_freq = length(frequencies_to_suppress);

notch_b = zeros(num_freq, 3); 
notch_a = zeros(num_freq, 3); 
notch_x_bufs = zeros(3, num_freq); 
notch_y_bufs = zeros(3, num_freq); 

for i = 1:num_freq
    f0 = frequencies_to_suppress(i);
    w0 = 2 * pi * f0 / fs;
    r = 0.993;  
    notch_b(i,:) = [1, -2*cos(w0), 1];
    notch_a(i,:) = [1, -2*r*cos(w0), r^2];
end

N = length(external_noise);
enhancedSpeech = zeros(N, 1);
externalNoiseEst = zeros(N, 1);

for i = 1:N
    x_sample = external_noise(i);
    y_sample = noisySpeech(i);
    
    % STEP 1: Notch filtering on external noise
    if preserveTonalNoise
        filtered_ref = x_sample;
        
        for j = 1:num_freq
            notch_x_bufs(:,j) = [filtered_ref; notch_x_bufs(1:2,j)];

            y_ref = notch_b(j,1)*notch_x_bufs(1,j) + notch_b(j,2)*notch_x_bufs(2,j) + notch_b(j,3)*notch_x_bufs(3,j) - notch_a(j,2)*notch_y_bufs(1,j) - notch_a(j,3)*notch_y_bufs(2,j);

            notch_y_bufs(:,j) = [y_ref; notch_y_bufs(1:2,j)];
            filtered_ref = y_ref;
        end
        x_filtered = filtered_ref;
    else
        % No notch filtering
        x_filtered = x_sample;
    end
    
    % STEP 2: RLS adaptive filtering
    rls_xn = [x_filtered; rls_xn(1:L-1)];
    noise_est = rls_w' * rls_xn;
    e = y_sample - noise_est;
    zn = rls_P * rls_xn;
    gn = zn / (lambda + rls_xn' * zn);
    rls_P = (rls_P - gn * rls_xn' * rls_P) / lambda;
    rls_w = rls_w + gn * e;

    enhancedSpeech(i) = e;
end

sound(enhancedSpeech,fs);

% Calculate SNR
initial_noise = (noisySpeech- cleanSpeech);
snr_initial = calculate_snr(cleanSpeech,initial_noise);
fprintf('\nInitial SNR: %.3f dB\n', snr_initial);
noise_residual = enhancedSpeech - cleanSpeech;
snr_output = calculate_snr(cleanSpeech,noise_residual);
fprintf('Final SNR: %.3f dB\n', snr_output);
fprintf('SNR Improvement: %.3f dB\n', snr_output - snr_initial);

if preserveTonalNoise
% Non Tonal-Measure in Partial Suppression
residual_tonal = extract_tonal_components(noise_residual, frequencies_to_suppress, fs);
nonTonal_residual = noise_residual - residual_tonal;
nonTonal_snr = calculate_snr(cleanSpeech,nonTonal_residual);
fprintf('Non-tonal SNR: %.3f dB\n', nonTonal_snr);

% Tonal Measure in Partial Suppression
[tpr] = calculate_tpr(noisySpeech, enhancedSpeech, frequencies_to_suppress, fs);
fprintf('Tonal Frequency Preservation Ratio: %.4f\n', tpr);
end

% Time domain plots
figure('Name', 'Time Domain', 'NumberTitle', 'off');
subplot(3,1,1); plot(cleanSpeech); title('Clean Speech');
subplot(3,1,2); plot(noisySpeech); title('Noisy Speech');
subplot(3,1,3); plot(enhancedSpeech); title('Enhanced Speech');

% Spectrogram comparison
figure('Name', 'Spectrogram Comparison', 'NumberTitle', 'off');
subplot(2,1,1);
spectrogram(cleanSpeech, 512, 256, 1024, fs, 'yaxis');
title('Clean Speech'); clim([-80 20]);
subplot(2,1,2);
spectrogram(enhancedSpeech, 512, 256, 1024, fs, 'yaxis');
title('Enhanced Speech'); clim([-80 20]);

function tonal_components = extract_tonal_components(input_signal, frequencies, fs)
    N = length(input_signal);
    tonal_components = zeros(N, 1);
    
    for freq_idx = 1:length(frequencies)
        f0 = frequencies(freq_idx);
        w0 = 2 * pi * f0 / fs;
        r = 0.993; 
        b = [1, -2*cos(w0), 1];
        a = [1, -2*r*cos(w0), r^2];
        notched_signal = filter(b, a, input_signal);
        tonal_components = tonal_components + (input_signal - notched_signal);
    end
end

function [tpr] = calculate_tpr(noisySpeech, enhancedSpeech, frequencies, fs)
    tonal_noisy = extract_tonal_components(noisySpeech, frequencies, fs);
    tonal_enhanced = extract_tonal_components(enhancedSpeech, frequencies, fs);

    tonal_noisy_energy = sum(tonal_noisy.^2);
    tonal_enhanced_energy = sum(tonal_enhanced.^2);

    tpr = tonal_enhanced_energy / tonal_noisy_energy;
end

function snr = calculate_snr(signal, noise)
    signal_power = sum(signal.^2) / length(signal);
    noise_power = sum(noise.^2) / length(noise);
    snr = 10*log10(signal_power / noise_power);
end