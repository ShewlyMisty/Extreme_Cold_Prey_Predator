clc;
clear;
global D1 D2 a b l d c0 c1 d1 mu cs Xend Tend m n subsetInterval noise_type noise_amplitude

%% Model Parameters
Xend = 5;
m = 2000; 
Tend = 5;
n = 500;
subsetInterval = 2;

D1 = 1; % Diffusion coefficient for species 1
D2 = 1; % Diffusion coefficient for species 2

a = 5; 
l = a; 
d = 3;
b = a + 1 - 1/sqrt(d);
c0 = b + sqrt(b^2 - 4*a);
mu = 1/2 * sqrt(b*sqrt(b^2 - 4*a) + b^2 - 2*a);
cs = 1/a * mu * (-b^2 + b*sqrt(b^2 - 4*a) + 3*a);

%% Spatial and Time Domains
x = linspace(0, Xend, m);
t = linspace(0, Tend, n);

%% Noise Setup and Baseline (Noiseless) Solution
noise_amplitude = 0.05; % Adjust as needed
noise_types = {'Gaussian','Uniform','Levy'}; % Noise types to compare

noise_type = 'None'; % No noise
options = odeset('RelTol',1e-6, 'AbsTol',1e-9);
baseline_sol = pdepe(0, @pde_s, @pdeic_s, @pdebc_s, x, t, options);
u_baseline = baseline_sol(:,:,1);

%% 1. 3D Surface Plot for Positive Lévy Noise
rng(42); % Fix seed for reproducibility
noise_type = 'Levy';
noisy_sol = pdepe(0, @pde_s, @pdeic_s, @pdebc_s, x, t, options);
u_noisy = noisy_sol(:,:,1);

figure;
surf(x(1:subsetInterval:end), t(1:subsetInterval:end), u_noisy(1:subsetInterval:end, 1:subsetInterval:end), 'EdgeColor', 'none');
colormap(jet);
colorbar;
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$t$', 'Interpreter', 'latex');
zlabel('$\hat{u}(t,x)$', 'Interpreter', 'latex');
view(-30,30);
light('Position',[1 0 1],'Style','infinite');
lighting gouraud;
alpha(0.9);

%% 2. Noise Type Comparison
fprintf('Comparing Gaussian, Uniform, and Positive Lévy noises...\n');
MSE = zeros(1, length(noise_types));
MaxDeviation = zeros(1, length(noise_types));

for i = 1:length(noise_types)
    noise_type = noise_types{i};
    rng(42); % Fix seed for reproducibility
    fprintf('Simulating with %s noise...\n', noise_type);
    noisy_sol = pdepe(0, @pde_s, @pdeic_s, @pdebc_s, x, t, options);
    u_noisy = noisy_sol(:,:,1);
    MSE(i) = mean((u_noisy(:)-u_baseline(:)).^2);
    MaxDeviation(i) = max(abs(u_noisy(:)-u_baseline(:)));
end

disp(table(noise_types', MSE', MaxDeviation', 'VariableNames', {'NoiseType','MSE','MaxDeviation'}));

%% Additional Surface Plots of the Exact Solution
[X, T] = meshgrid(x, t);
Z = c0*(1 + 1./(-3/2 + 1/2*tanh(mu/2*(X - cs*T))));

figure;
surf(x(1:subsetInterval:end), t(1:subsetInterval:end), Z(1:subsetInterval:end, 1:subsetInterval:end), 'EdgeColor', 'none');
colormap(jet);
colorbar;
xlabel('$x$', 'Interpreter', 'latex');
ylabel('$t$', 'Interpreter', 'latex');
zlabel('$u(t,x)$', 'Interpreter', 'latex');
view(-30,30);
light('Position',[1 0 1],'Style','infinite');
lighting gouraud;
alpha(0.9);
%% PDE Function Definitions
function [c,f,s] = pde_s(x,t,u,DuDx)
    global D1 D2 a b l d
    c = [1; 1];
    f = [D1*DuDx(1); D2*DuDx(2)];
    s = [-a*u(1) + (1+a)*u(1)^2 - u(1)^3 - u(1)*u(2); 
         b*u(1)*u(2) - l*u(2) - d*u(2)^3];
end

function [pl,ql,pr,qr] = pdebc_s(xl,ul,xr,ur,t)
    global c0 cs mu d
    % Left boundary
    pl = [ ul(1) - ( c0*(1+1/(-3/2+1/2*tanh(mu/2*(0-cs*t))) ) ) ; 
           ul(2) - (1/sqrt(d))*( c0*(1+1/(-3/2+1/2*tanh(mu/2*(0-cs*t))) ) ) ];
    ql = [0;0];
    % Right boundary
    pr = [ ur(1) - ( c0*(1+1/(-3/2+1/2*tanh(mu/2*(5-cs*t))) ) ) ; 
           ur(2) - (1/sqrt(d))*( c0*(1+1/(-3/2+1/2*tanh(mu/2*(5-cs*t))) ) ) ];
    qr = [0;0];
end

function u_ic = pdeic_s(x)
    global c0 cs mu d noise_type noise_amplitude
    % Exact initial condition
    u0 = c0*(1 + 1./(-3/2 + 1/2*tanh(mu/2*(x - cs*0))));
    v0 = 1/sqrt(d)*(c0*(1 + 1./(-3/2 + 1/2*tanh(mu/2*(x - cs*0)))));
    
    % Add noise based on selected type
    switch noise_type
        case 'None'
            noise = 0;
        case 'Gaussian'
            noise = noise_amplitude * randn(size(x));
        case 'Uniform'
            noise = noise_amplitude * (2*rand(size(x))-1);
        case 'Levy'
            noise = levy_noise_positive(1.5, noise_amplitude, size(x));  % Assumes you have this function
        otherwise
            error('Unknown noise type: %s', noise_type);
    end
    
    u0 = u0 + noise;
    v0 = v0 + noise;
    u_ic = [u0; v0];
end
