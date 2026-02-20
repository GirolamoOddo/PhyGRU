% run in MATLAB R2021b+ with Deep Learning Toolbox
clear; clc; rng(0);

useGPU = false;
BATCH = 16; N_SAMPLES = 256; EPOCHS = 20; LR = 1e-3;
T = 10; input_dim = 1; phys_state_dim = 2; phys_latent_dim = 3;
gru_hidden_dim = 32; output_dim = 1; dt = 0.01;

device = 'cpu'; if useGPU && canUseGPU(), device = 'gpu'; end

% ground-truth physics for data generation
physics_gt.a = 1.0; physics_gt.b = 0.5; physics_gt.c = 0.2;

% synthetic data
all_u = randn(N_SAMPLES, T, input_dim,'single') * 2.0;
all_y = simulate_physics_batch(all_u, physics_gt, dt, phys_state_dim);

% initialize params
params = initializeModelParams(phys_state_dim, phys_latent_dim, input_dim, ...
    gru_hidden_dim, output_dim);
physicsParams.a = dlarray(single(0.5)); physicsParams.b = dlarray(single(0.6)); physicsParams.c = dlarray(single(0.7));

if strcmp(device,'gpu')
    params = toGPU(params);
    physicsParams = structfun(@gpuArray, physicsParams, 'UniformOutput', false);
end

% Adam states
iteration = 0; beta1=0.9; beta2=0.999; eps=1e-8;
trailingAvg = initAdamState(params); trailingAvgSq = initAdamState(params);
trailingAvgP = initAdamState(physicsParams); trailingAvgSqP = initAdamState(physicsParams);

for epoch = 1:EPOCHS
    idx = randperm(N_SAMPLES); epochLoss = 0;
    for bStart = 1:BATCH:N_SAMPLES
        iteration = iteration + 1;
        bEnd = min(bStart+BATCH-1, N_SAMPLES); batchIdx = idx(bStart:bEnd);
        xb = all_u(batchIdx,:,:); yb = all_y(batchIdx,:,:);
        if strcmp(device,'gpu'), xb = gpuArray(xb); yb = gpuArray(yb); end
        dlX = dlarray(single(xb)); dlY = dlarray(single(yb));

        [gradP, gradPhys, loss] = dlfeval(@modelGradients, params, physicsParams, dlX, dlY, dt, ...
            phys_state_dim, phys_latent_dim, input_dim, gru_hidden_dim, output_dim);

        epochLoss = epochLoss + double(gather(extractdata(loss))) * numel(batchIdx);

        % update params
        names = fieldnames(params);
        for i=1:numel(names)
            n = names{i};
            g = gradP.(n);
            [params.(n), trailingAvg.(n), trailingAvgSq.(n)] = ...
                adamupdate(params.(n), g, trailingAvg.(n), trailingAvgSq.(n), iteration, LR, beta1, beta2, eps);
        end
        pnames = fieldnames(physicsParams);
        for i=1:numel(pnames)
            n = pnames{i};
            g = gradPhys.(n);
            [physicsParams.(n), trailingAvgP.(n), trailingAvgSqP.(n)] = ...
                adamupdate(physicsParams.(n), g, trailingAvgP.(n), trailingAvgSqP.(n), iteration, LR, beta1, beta2, eps);
        end
    end
    epochLoss = epochLoss / N_SAMPLES;
    if mod(epoch,5)==0 || epoch==1 || epoch==EPOCHS
        fprintf("Epoch %d/%d    Loss: %.6f\n", epoch, EPOCHS, epochLoss);
    end
end

save('phygru_hybrid_example.mat','params','physicsParams','input_dim','phys_state_dim','phys_latent_dim','gru_hidden_dim','output_dim','dt');
disp('Model saved to phygru_hybrid_example.mat');

% reload and inference
S = load('phygru_hybrid_example.mat'); params2 = S.params; physicsParams2 = S.physicsParams;
if strcmp(device,'gpu'), params2 = toGPU(params2); physicsParams2 = structfun(@gpuArray, physicsParams2,'UniformOutput',false); end

test_u = all_u(1:8,:,:); test_y_true = all_y(1:8,:,:);
if strcmp(device,'gpu'), test_u = gpuArray(test_u); test_y_true = gpuArray(test_y_true); end
dlXtest = dlarray(single(test_u));
pred_test = predictSequence(params2, physicsParams2, dlXtest, dt, phys_state_dim, phys_latent_dim, input_dim, gru_hidden_dim, output_dim);
pred_test = gather(extractdata(pred_test));

disp("Test pred shape:"); disp(size(pred_test));
disp("Test true shape:"); disp(size(test_y_true));
fprintf("First-sample first-timestep true vs pred: %.6f vs %.6f\n", test_y_true(1,1,1), pred_test(1,1,1));

%% helpers 

function traj = simulate_physics_batch(u_all, physics, dt, phys_state_dim)
N = size(u_all,1); T = size(u_all,2); traj = zeros(N,T,1,'single'); state = zeros(N,phys_state_dim,'single');
for t=1:T
    u_t = squeeze(u_all(:,t,:));
    state = state + dt * secondOrderLaw_forward(state, u_t, physics);
    traj(:,t,1) = state(:,1);
end
end

function out = secondOrderLaw_forward(state, u, physics)
x = state(:,1); xd = state(:,2); u_s = reshape(u(:,1),[],1);
a = physics.a; b = physics.b; c = physics.c;
xdd = (u_s - b .* xd - c .* x) ./ (a + 1e-12);
out = [xd, xdd];
end

function params = initializeModelParams(state_dim, latent_dim, input_dim, gru_hidden_dim, output_dim)
total_state = state_dim + latent_dim;
if latent_dim>0
    in_lat = state_dim + latent_dim + input_dim;
    params.latent_W = dlarray(single(initializeGlorot([latent_dim, in_lat])));
    params.latent_b = dlarray(zeros(latent_dim,1,'single'));
    params.r_W = dlarray(single(initializeGlorot([latent_dim, in_lat])));
    params.r_b = dlarray(zeros(latent_dim,1,'single'));
else
    params.latent_W=[]; params.latent_b=[]; params.r_W=[]; params.r_b=[];
end
in_z = total_state + input_dim;
params.z_W = dlarray(single(initializeGlorot([total_state, in_z])));
params.z_b = dlarray(zeros(total_state,1,'single'));

params.gru_Wz = dlarray(single(initializeGlorot([gru_hidden_dim, total_state])));
params.gru_Uz = dlarray(single(initializeGlorot([gru_hidden_dim, gru_hidden_dim])));
params.gru_bz = dlarray(zeros(gru_hidden_dim,1,'single'));

params.gru_Wr = dlarray(single(initializeGlorot([gru_hidden_dim, total_state])));
params.gru_Ur = dlarray(single(initializeGlorot([gru_hidden_dim, gru_hidden_dim])));
params.gru_br = dlarray(zeros(gru_hidden_dim,1,'single'));

params.gru_Wh = dlarray(single(initializeGlorot([gru_hidden_dim, total_state])));
params.gru_Uh = dlarray(single(initializeGlorot([gru_hidden_dim, gru_hidden_dim])));
params.gru_bh = dlarray(zeros(gru_hidden_dim,1,'single'));

params.fnn_W1 = dlarray(single(initializeGlorot([64, gru_hidden_dim])));
params.fnn_b1 = dlarray(zeros(64,1,'single'));
params.fnn_W2 = dlarray(single(initializeGlorot([output_dim, 64])));
params.fnn_b2 = dlarray(zeros(output_dim,1,'single'));
end

function W = initializeGlorot(sz)
fan_in = sz(2); fan_out = sz(1);
limit = sqrt(6 / (fan_in + fan_out));
W = (rand(sz,'single') * 2 - 1) * limit;
end

function s = initAdamState(params)
names = fieldnames(params); for i=1:numel(names), s.(names{i}) = zeros(size(params.(names{i})), 'like', params.(names{i})); end
end

function p = toGPU(p)
names = fieldnames(p); for i=1:numel(names), if ~isempty(p.(names{i})), p.(names{i}) = gpuArray(p.(names{i})); end, end
end

function [gradientsParams, gradientsPhysics, loss] = modelGradients(params, physicsParams, dlX, dlY, dt, state_dim, latent_dim, input_dim, gru_hidden_dim, output_dim)
pred = predictSequence(params, physicsParams, dlX, dt, state_dim, latent_dim, input_dim, gru_hidden_dim, output_dim);
loss = mean((pred - dlY).^2,'all');
pnames = fieldnames(params); for i=1:numel(pnames), gradientsParams.(pnames{i}) = dlgradient(loss, params.(pnames{i})); if isempty(gradientsParams.(pnames{i})), gradientsParams.(pnames{i}) = zeros(size(params.(pnames{i})), 'like', params.(pnames{i})); end, end
pn = fieldnames(physicsParams); for i=1:numel(pn), gradientsPhysics.(pn{i}) = dlgradient(loss, physicsParams.(pn{i})); if isempty(gradientsPhysics.(pn{i})), gradientsPhysics.(pn{i}) = zeros(size(physicsParams.(pn{i})), 'like', physicsParams.(pn{i})); end, end
end

function dlYpred = predictSequence(params, physicsParams, dlX, dt, state_dim, latent_dim, input_dim, gru_hidden_dim, output_dim)
B = size(dlX,1); T = size(dlX,2); total_state = state_dim + latent_dim;
state = dlarray(zeros(B,total_state,'single'));
outputs = dlarray(zeros(B,T,total_state,'single'));
for t=1:T
    u_t = reshape(dlX(:,t,:),B,input_dim);
    state = phyGRUCellForward(state, u_t, params, physicsParams, state_dim, latent_dim, input_dim, dt);
    outputs(:,t,:) = state;
end
h = dlarray(zeros(B,gru_hidden_dim,'single'));
Ypred = dlarray(zeros(B,T,output_dim,'single'));
for t=1:T
    x_t = reshape(outputs(:,t,:),B,total_state);
    z_t = sigmoid(x_t * params.gru_Wz' + h * params.gru_Uz' + params.gru_bz');
    r_t = sigmoid(x_t * params.gru_Wr' + h * params.gru_Ur' + params.gru_br');
    h_tilde = tanh(x_t * params.gru_Wh' + ( (r_t .* h) * params.gru_Uh' ) + params.gru_bh');
    h = (1 - z_t) .* h + z_t .* h_tilde;
    fc1 = max(h * params.fnn_W1' + params.fnn_b1', 0);
    y_t = fc1 * params.fnn_W2' + params.fnn_b2';
    Ypred(:,t,:) = reshape(y_t, B, 1, []);
end
dlYpred = Ypred;
end

function next_state = phyGRUCellForward(hx, u, params, physicsParams, state_dim, latent_dim, input_dim, dt)
B = size(hx,1);
phys = hx(:,1:state_dim);
if latent_dim>0, latent = hx(:, state_dim+1:end); else latent = dlarray(zeros(B,0,'single')); end
x = phys(:,1); xd = phys(:,2); u_s = reshape(u(:,1),[],1);
a = physicsParams.a; b = physicsParams.b; c = physicsParams.c;
xdd = (u_s - b .* xd - c .* x) ./ (a + 1e-12);
phys_dot = [xd, xdd];
phys_next = phys + dt * phys_dot;
if latent_dim>0
    in_lat = [phys, latent, u];
    r = sigmoid(in_lat * params.r_W' + params.r_b');
    gated_latent = r .* latent;
    latent_input = [phys, gated_latent, u];
    latent_dot = latent_input * params.latent_W' + params.latent_b';
    latent_next = latent + dt * latent_dot;
    candidate = [phys_next, latent_next];
else
    candidate = phys_next;
end
z = sigmoid([hx, u] * params.z_W' + params.z_b');
next_state = z .* candidate + (1 - z) .* hx;
end

function y = sigmoid(x), y = 1 ./ (1 + exp(-x)); end
