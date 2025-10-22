%% Short script for computing t-scores for block design experiment
load("/mnt/storage/rexfung/20251003tap/recon/mslr/tmpAvgInit/6x_5e-3.mat") % X
Nt = size(X,ndims(X));

%% parameters
TR = 0.8; % s
task_period = 40; % s
T = round(task_period / TR); % frames
TF = -square(2*pi*(1:Nt)/T); % task function
HRF = spm_hrf(TR); % haemodynamic response function
task_response = conv(TF, HRF, 'same');

%% t-scores 
GLM = [ones(1, Nt); 0:Nt-1; task_response]';
c = [0, 0, 1]'; % 3rd regressor is BOLD
t_img = tscores(X, GLM, c);

%% flip data around to canonical view
t_img = flip(flip(t_img, 2), 3);
X = flip(flip(X, 2), 3);

%% Viszualize axial and coronal slices
tlim = [4, 15];
cmap = parula;

figure;
qt(t_img), abs(X(:,:,:,1)), tlim, cmap, 'MSLR recon of 6x undersampled data');

figure;
qt(permute(t_img, [1 3 2]), permute(abs(X(:,:,:,1)), [1 3 2]), tlim, cmap, 'MSLR recon of 6x undersampled data');
return;

%% Do the same for product sequence
X = niftiread("/mnt/storage/rexfung/20251003tap/recon/product/6x.nii");
X = X(:,:,:,13:end); % discard the first 9.6 seconds
X = double(X);

% t-scores 
t_img = tscores(X, GLM, c);

% flip data around to canonical view
t_img = flip(flip(t_img, 2), 3);
X = flip(flip(X, 2), 3);

% Viszualize axial and coronal slices
tlim = [4, 15];
cmap = parula;

figure;
qt(t_img, abs(X(:,:,:,1)), tlim, cmap, 'product (SMS-EPI)');

figure;
qt(permute(t_img, [1 3 2]), permute(abs(X(:,:,:,1)), [1 3 2]), tlim, cmap, 'product (SMS-EPI)');