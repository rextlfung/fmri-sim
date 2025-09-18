%% Short script for simulating retrospective undersampling and recon
verbose = 0;
datdir = "/mnt/storage/rexfung/20250902sim/";

load(datdir + "rest_sense.mat") % I_sense
load(datdir + "2d.mat") % ksp_2d_epi
I_sense = flip(I_sense,2);
I_sense = I_sense(:,:,:,13:end); % Discard frames prior to steady state
[Nx,Ny,Nz,Nt] = size(I_sense);
Nc = size(ksp_2d_epi, 4);

%% Make sensitivity maps
fn_smaps = datdir + "smaps.mat";
if ~isfile(fn_smaps)
    etl = size(ksp_2d_epi,2);
    ksp = ifftshift(fft(fftshift(ksp_2d_epi),Nz,3));
    ksp_zf = zeros(Nx,Ny,Nz,Nc);
    ksp_zf(:,(end-etl+1):end,:,:) = ksp;
    [smaps, emaps] = makeSmaps(ksp_zf, 'pisco');
    save(fn_smaps, 'smaps', 'emaps', '-v7.3');
    smaps_new = smaps;
elseif isfile(fn_smaps)
    load(fn_smaps);
    smaps_new = smaps;
end

%% Extact one slice and rescale to [0, 1]
z = 40;
gt = squeeze(I_sense(:,:,z,:));
smaps = squeeze(smaps_new(:,:,z,:));

%% Create regions of interest
M_mask = false(Nx, Ny);
M_mask(31:60, 36:55) = permute(blockMmask(20,30), [2 1]);

U_mask = false(Nx, Ny);
U_mask(66:85, 61:85) = permute(blockUmask(25,20), [2 1]);

circ_mask = false(Nx, Ny);
circ_mask(34:58, 59:83) = circleMask(25,25);

if verbose
    figure; jim((M_mask + circ_mask) .* gt(:,:,1))
end

%% Synthesize BOLD activation with amplitude equal to ~3% signal of each ROI
TR = 0.8; % s
task_period = 40; % s
T = round(task_period / TR); % frames
TF = square(2*pi*(1:Nt)/T); % task function
HRF = spm_hrf(TR); % haemodynamic response function
amp = 0.5e-2; % amplitude of BOLD activation relative to mean signal
task_response = conv(TF, HRF, 'same');
BOLD = amp*task_response/max(task_response);

img = gt;
img = img .* (ones(Nx,Ny,Nt) + single(M_mask).*reshape(BOLD, 1, 1, Nt));
img = img .* (ones(Nx,Ny,Nt) + single(circ_mask).*reshape(BOLD, 1, 1, Nt));

if verbose
    interactive3D(abs(img))
    interactive3D(angle(img))
end

%% Generate multicoil images
img_mc = zeros(Nx, Ny, Nc, Nt);
for t = 1:Nt
    img_mc(:,:,:,t) = img(:,:,t) .* smaps;
end

if verbose
    interactive4D(abs(img_mc))
end

%% Generate k-space images
ksp_mc = ifftshift(fft(fft(fftshift(img_mc), Nx, 1), Ny, 2));

if verbose
    interactive4D(log(abs(ksp_mc) + eps))
end

%% Generate random or standard caipi sampling masks
omegas = zeros(Nx, Ny, Nt);

mode = 'caipi';
switch mode
    case 'caipi'
        Rx = 2; Ry = 3;
        caipi = 3;
        omegas(1:Rx:Nx, 1:Ry:Ny, :) = 1;

        x_idx = 1:Rx:Nx;
        for i = 1:length(x_idx)
            ix = x_idx(i);
            shift = mod(i - 1, caipi);
            omegas(ix,:,:) = circshift(omegas(ix,:,:), shift, 2);
        end

    case 'rand'
        Rx = sqrt(3); Ry = sqrt(3); R = [Rx Ry];
        caipi = 2;
        acs_x = 0.2; acs_y = 0.2; acs = [acs_x acs_y];
        weights_x = normpdf(1:Nx, mean(1:Nx), Nx/6);
        weights_y = normpdf(1:Ny, mean(1:Ny), Ny/6);
        max_kx_step = round(Nx/16);
        
        for t = 1:Nt
            rng(t);
            [omegas(:,:,t), ~, ~]= randsamp2dcaipi([Nx, Ny], R, acs, weights_x, weights_y, max_kx_step, caipi);
        end
end

if verbose
    interactive3D(omegas)
end

%% Undersample
ksp_us = ksp_mc .* reshape(omegas, Nx, Ny, 1, Nt);

%% zero_filled IFT images
img_zf = ifftshift(ifft2(fftshift(ksp_us)));

%% rsos combination
rec_rsos = zeros(Nx, Ny, Nt);
for t = 1:Nt
    rec_rsos(:,:,t) = sqrt(sum(abs(img_zf(:,:,:,t)).^2, 3));
end

if verbose
    interactive3D(abs(rec_rsos))
end

%% SENSE combination
rec_sense = zeros(Nx, Ny, Nt);
for t = 1:Nt
    rec_sense(:,:,t) = sum(img_zf(:,:,:,t) .* conj(smaps), 3);
end

if verbose
    interactive3D(abs(rec_sense))
end

%% CG-SENSE recon
rec_cgs = zeros(Nx, Ny, Nt);
smaps_reshaped = reshape(smaps, [Nx, Ny, 1, Nc]);

parfor t = 1:Nt
    k_t = permute(ksp_us(:,:,:,t), [1 2 4 3]);
    rec_cgs(:,:,t) = bart('pics', k_t, smaps_reshaped);
end

if verbose
    interactive3D(abs(rec_cgs))
end

%% GLM
GLM = [ones(1, Nt); 0:Nt-1; BOLD]';
c = [0, 0, 1]'; % 3rd regressor is BOLD

%% t-scores
t_img = tscores(img, GLM, c);
t_rsos = tscores(rec_rsos, GLM, c);
t_sense = tscores(rec_sense, GLM, c);
t_cgs = tscores(rec_cgs, GLM, c);

%% Viszualize t-scores for simple cases
tlim = [3, 15];
cmap = parula;

figure; tiledlayout(1,4,"TileSpacing","tight");
nexttile; qt(t_img, abs(img(:,:,1)), tlim, cmap, 'Simulated task response + rs-fMRI');
nexttile; qt(t_rsos, abs(rec_rsos(:,:,1)), tlim, cmap, 'RSOS coil combination');
nexttile; qt(t_sense, abs(rec_sense(:,:,1)), tlim, cmap, 'SENSE coil combination');
nexttile; qt(t_cgs, abs(rec_cgs(:,:,1)), tlim, cmap, 'CG-SENSE');

%% L1-regularized recon
figure; tiledlayout(1,4,"TileSpacing","tight");
tlim = [3, 15];
cmap = parula;

for n = 2:5 % Loop over regularization strengths
    smaps_reshaped = reshape(smaps, [Nx, Ny, 1, Nc]);
    rec_l1 = zeros(Nx, Ny, Nt);
    parfor t = 1:Nt
        k_t = permute(ksp_us(:,:,:,t), [1 2 4 3]);
        cmd = sprintf('pics -l1 -r%f', 10^(-n));
        rec_l1(:,:,t) = bart(cmd, k_t, smaps_reshaped);
    end
    t_l1 = tscores(rec_l1, GLM, c);
    nexttile;
    plotname = sprintf('L1-regularized CGS, lambda = 1e-%d', n);
    qt(t_l1, abs(rec_l1(:,:,1)), tlim, cmap, plotname);
end

%% LLR recon (in Julia)
figure; tiledlayout(1,4,"TileSpacing","tight");
tlim = [3, 20];
cmap = parula;

for n = 2:5 % Loop over regularization strengths
    fn = datdir + sprintf('caipi_comp/rec_llr_5e-%d', n);
    load(fn);
    rec_lr = squeeze(X);
    t_lr = tscores(rec_lr, GLM, c);

    nexttile;
    plotname = sprintf('Multiscale LLR, lambda = 5e-%d', n);
    qt(t_lr, abs(rec_lr(:,:,1)), tlim, cmap, plotname);
end