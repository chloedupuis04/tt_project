clear; close all; clc;

d = 2;

% Initialize randomness only once
rng("shuffle");

%  Genz oscillatory function

bOsc = 284.6;
hOsc = 1.5;

% One random scalar shift
wOsc = rand();

% Random coefficients
cRawOsc = rand(1, d);

% Normalize the coefficients
cOsc = cRawOsc / sum(abs(cRawOsc)) * (bOsc / d^hOsc);

c1Osc = cOsc(1);
c2Osc = cOsc(2);

% Save oscillatory parameters
parametersOsc = table(wOsc, c1Osc, c2Osc);

% Change this path when running on another computer
outputFileOsc = "/Users/coco/Desktop/tt_project/tests/genz_oscillatory_parameters.csv";

writetable(parametersOsc, outputFileOsc);

% Define the oscillatory function
fGenzOscillatory = @(x,y) ...
    cos(2*pi*wOsc ...
    + c1Osc*(x+1)/2 ...
    + c2Osc*(y+1)/2);


%  Genz corner peak function

bCorner = 185.0;
hCorner = 2.0;

% Random coefficients
cRawCorner = rand(1, d);

% Normalize the coefficients
cCorner = cRawCorner / sum(abs(cRawCorner)) ...
          * (bCorner / d^hCorner);

c1Corner = cCorner(1);
c2Corner = cCorner(2);

% Save corner peak parameters
parametersCorner = table(c1Corner, c2Corner);

% Change this path if running on another computer
outputFileCorner = "/Users/coco/Desktop/tt_project/tests/genz_corner_peak_parameters.csv";

writetable(parametersCorner, outputFileCorner);

% Define the corner peak function
fGenzCornerPeak = @(x,y) ...
    (1.0 ...
    + c1Corner*(x+1)/2 ...
    + c2Corner*(y+1)/2).^(-(d+1));


fAckley = @(x,y) -20 .* exp(-0.2 .* sqrt((1/7) .* (x.^2 + y.^2))) - exp((1/7) .* (cos(2*pi*x) + cos(2*pi*y))) + 20 + exp(1);
% "Dictionary" of functions: name -> handle
funcNames    = { 'ackley' ,'genz_corner_peak','genz_oscillatory'};
funcHandles  = {fAckley,fGenzCornerPeak,fGenzOscillatory};

tol_list = [1e-6, 1e-8, 1e-10, 1e-12, 1e-14];
N_s = 1000;
N_t = 1000;
Ntol = numel(tol_list);

outDir = 'tests/figures';
if ~exist(outDir, 'dir'); mkdir(outDir); end
N_grid = 100;
tol = 1e-10;

for k = 1:numel(funcHandles)

    fname = funcNames{k};
    fRaw  = funcHandles{k};

    if  fname== "ackley"
        domain=[-32.768,32.768;-32.768,32.768];
    else 
    domain = [-1, 1;
              -1, 1];
    end 
    % Function expressed on the EFTT reference domain [-1,1]^2
    fMapped = @(z) fRaw( ...
        mapDim(z(:,1), domain(1,:)), ...
        mapDim(z(:,2), domain(2,:)));

    %% Construct EFTT for one fixed tolerance

    eftt = EFTT(fMapped, 2, 'tol', tol);

    d = degree(eftt);
    nChebX = d(1);
    nChebY = d(2);

    %% Regular grid in the true domain

    xGrid = linspace(domain(1,1), domain(1,2), N_grid);
    yGrid = linspace(domain(2,1), domain(2,2), N_grid);

    [X, Y] = ndgrid(xGrid, yGrid);

    %% Map grid to EFTT reference coordinates

    xRef = mapDimInv(xGrid, domain(1,:));
    yRef = mapDimInv(yGrid, domain(2,:));

    [Xref, Yref] = ndgrid(xRef, yRef);

    %% Evaluate EFTT interpolant

    K = reshape( ...
        evaluateBatch(eftt, [Xref(:), Yref(:)]), ...
        N_grid, N_grid);

    %% Evaluate true function

    Ftrue = fRaw(X, Y);

    %% Absolute and relative errors

    absError = abs(Ftrue - K);



    %% Plot interpolant

    fig = figure('Position', [100, 100, 700, 600]);

    surf(X, Y, K, 'EdgeColor', 'none');
    xlabel('x');
    ylabel('y');
    zlabel('interpolant value');
    title("EFTT interpolant: " + string(fname));
    colorbar;
    view(3);
    saveas(fig, fullfile(outDir, sprintf('3D_interpolation%s.pdf', fname)));
    close(fig);

    %% Plot true function

    fig = figure('Position', [100, 100, 700, 600]);

    surf(X, Y, Ftrue, 'EdgeColor', 'none');
    xlabel('x');
    ylabel('y');
    zlabel('function value');
    title("True function: " + string(fname));
    colorbar;
    view(3);
    saveas(fig, fullfile(outDir, sprintf('3D_fucntion%s.pdf', fname)));
    close(fig);


    fig = figure('Position', [100, 100, 700, 600]);

    surf(X, Y, absError, 'EdgeColor', 'none');
    xlabel('x');
    ylabel('y');
    zlabel('absolute error');
    title("Absolute error: " + string(fname));
    colorbar;
    view(3);
    saveas(fig, fullfile(outDir, sprintf('3D_error%s.pdf', fname)));
    close(fig);

end

for k = 1:numel(funcHandles)

    fname = funcNames{k};
    fRaw  = funcHandles{k};
    domain = [-1, 1; -1, 1];   % row1 = x bounds, row2 = y bounds
    fMapped = @(z) fRaw(mapDim(z(:,1), domain(1,:)), mapDim(z(:,2), domain(2,:)));

    err     = zeros(1, Ntol);
    nChebX  = zeros(1, Ntol);
    nChebY  = zeros(1, Ntol);

     % Random test points in the true domain
     xPts = domain(1,1) + (domain(1,2)-domain(1,1)) * rand(N_s, 1);
     yPts = domain(2,1) + (domain(2,2)-domain(2,1)) * rand(N_t, 1);
     xRef = mapDimInv(xPts, domain(1,:));
     yRef = mapDimInv(yPts, domain(2,:));

    for i = 1:Ntol
        tol = tol_list(i);
        eftt = EFTT(fMapped, 2, 'tol', tol);
        d = degree(eftt);
        nChebX(i) = d(1);
        nChebY(i) = d(2);
        [XX, YY] = ndgrid(xRef, yRef);
        K = reshape(evaluateBatch(eftt, [XX(:), YY(:)]), N_s, N_t);
        [XXtrue, YYtrue] = ndgrid(xPts, yPts);
        Kexact = fRaw(XXtrue, YYtrue);
        err(i) = norm(K - Kexact, 'fro') / norm(Kexact, 'fro');
    end

    figX = figure;
    semilogy(nChebX, err, '-o');
    xlabel('nbr cheb nodes (x)');
    ylabel('Relative error');
    title(sprintf('EFTT relative error vs. nbr cheb nodes in x (%s)', fname), 'Interpreter', 'none');
    grid on;
    saveas(figX, fullfile(outDir, sprintf('error_EFTT_vs_nx_%s.pdf', fname)));
    close(figX);

    figY = figure;
    semilogy(nChebY, err, '-o');
    xlabel('nbr cheb nodes (y)');
    ylabel('Relative error');
    title(sprintf('EFTT relative error vs. nbr cheb nodes in y (%s)', fname), 'Interpreter', 'none');
    grid on;
    saveas(figY, fullfile(outDir, sprintf('error_EFTT_vs_ny_%s.pdf', fname)));
    close(figY);

    figX=figure;
    semilogx(tol_list,nChebX);
    xlabel('tol');
    ylabel('nbr cheb nodes');
    title(sprintf('EFTT nbr cheb nodes in x vs. tol (%s)', fname), 'Interpreter', 'none');
    grid on;
    saveas(figX, fullfile(outDir, sprintf('nx_EFTT_vs_tol_%s.pdf', fname)));
    close(figX);

    figY=figure;
    semilogx(tol_list,nChebY);
    xlabel('tol');
    ylabel('nbr cheb nodes');
    title(sprintf('EFTT nbr cheb nodes in y vs. tol (%s)', fname), 'Interpreter', 'none');
    grid on;
    saveas(figY, fullfile(outDir, sprintf('ny_EFTT_vs_tol_%s.pdf', fname)));
    close(figY);

    resultsTable = table( ...
    tol_list(:), ...
    nChebX(:), ...
    nChebY(:), ...
    err(:), ...
    'VariableNames', ...
    {'tol', 'nChebX', 'nChebY', 'relativeError'} ...
    );

    writetable( ...
    resultsTable, ...
    fullfile("/Users/coco/Desktop/tt_project/tests", sprintf('EFTT_results_%s.csv', fname)) ...
    );
end




%% Test function: f(x,y) = 1/|x-y|
% fRaw = @(x,y) 1 ./ abs(x - y);

% %% Parameters
% S        = [1e-2, 1e-3, 1e-4, 1e-6];
% tol_list = [1e-6, 1e-8, 1e-10, 1e-12,1e-14];
% N_s = 1000;
% N_t = 1000;
% Ns   = numel(S);
% Ntol = numel(tol_list);
% 
% err     = zeros(Ns, Ntol);
% nChebX  = zeros(Ns, Ntol);
% nChebY  = zeros(Ns, Ntol);
% 
% outDir = 'tests/figures';
% if ~exist(outDir, 'dir'); mkdir(outDir); end
% 
% figX = figure; hold on;
% figY = figure; hold on;
% 
% for iS = 1:Ns
%     s = S(iS);
%     domain = [0, 1 - s/2; 1 + s/2, 2];   % row1 = x bounds, row2 = y bounds
%     fMapped = @(z) fRaw( mapDim(z(:,1), domain(1,:)), mapDim(z(:,2), domain(2,:)) );
% 
%     fprintf('Building EFTT for s = %.0e ...\n', s);
% 
%     for i = 1:Ntol
%         tol = tol_list(i);
%         eftt = EFTT(fMapped, 2, 'tol', tol);
% 
%         d = degree(eftt);          % assign first, then index
%         nChebX(iS,i) = d(1);
%         nChebY(iS,i) = d(2);
% 
%         % Random test points in the true domain
%         xPts = domain(1,1) + (domain(1,2)-domain(1,1)) * rand(N_s, 1);
%         yPts = domain(2,1) + (domain(2,2)-domain(2,1)) * rand(N_t, 1);
%         xRef = mapDimInv(xPts, domain(1,:));
%         yRef = mapDimInv(yPts, domain(2,:));
% 
%         [XX, YY] = ndgrid(xRef, yRef);
%         K = reshape(evaluateBatch(eftt, [XX(:), YY(:)]), N_s, N_t);
%         Kexact = 1 ./ abs(xPts - yPts');
% 
%         err(iS,i) = norm(K - Kexact, 'fro') / norm(Kexact, 'fro');
%     end
% 
%     figure(figX);
%     semilogy(nChebX(iS,:), err(iS,:), '-o', 'DisplayName', sprintf('s = %.0e', s));
% 
%     figure(figY);
%     semilogy(nChebY(iS,:), err(iS,:), '-o', 'DisplayName', sprintf('s = %.0e', s));
% end
% 
% figure(figX);
% xlabel('nbr cheb nodes (x)');
% ylabel('Relative error');
% title('EFTT relative error vs. nbr cheb nodes in x');
% legend show; grid on; hold off;
% saveas(figX, fullfile(outDir, 'error_EFTT_vs_nx.pdf'));
% 
% figure(figY);
% xlabel('nbr cheb nodes (y)');
% ylabel('Relative error');
% title('EFTT relative error vs. nbr cheb nodes in y');
% legend show; grid on; hold off;
% saveas(figY, fullfile(outDir, 'error_EFTT_vs_ny.pdf'));

%% ---------------------------------------------------------------------
%  Helper functions
%  ---------------------------------------------------------------------
function y = mapDim(xi, bounds)
    y = bounds(1) + (bounds(2)-bounds(1)) .* (xi+1)/2;
end

function xi = mapDimInv(x, bounds)
    xi = 2*(x - bounds(1))/(bounds(2)-bounds(1)) - 1;
end

function vals = evaluateBatch(eftt, pts)
    n = size(pts, 1);
    try
        vals = eftt.fevaluate(pts);
        vals = vals(:);
        if numel(vals) ~= n
            error('evaluateBatch:sizeMismatch', 'batch eval size mismatch');
        end
    catch
        vals = zeros(n, 1);
        for k = 1:n
            vals(k) = eftt.fevaluate(pts(k, :));
        end
    end
end