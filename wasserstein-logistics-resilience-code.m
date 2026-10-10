
clear; clc; close all;


OUT_DIR = fullfile(pwd, "results");
FIG_DIR = fullfile(OUT_DIR, "figures");
TAB_DIR = fullfile(OUT_DIR, "tables");
DAT_DIR = fullfile(OUT_DIR, "data");
ensure_dir(FIG_DIR); ensure_dir(TAB_DIR); ensure_dir(DAT_DIR);


XY = [ ...
    10 80; 20 60; 30 75; 40 40; 50 70; ...
    60 20; 70 50; 80 30; 25 20; 35 10; ...
    55 85; 75 75; 85 55; 15 35; 65 60];

nNodes = size(XY,1);
supplyNodes = 1:5;
demandNodes = 11:15;


S0 = [120 100 90 110 80]';                 
D0 = [130 95 85 100 90]';                  

T = 10;                                     


dBase = pairwise_dist(XY(supplyNodes,:), XY(demandNodes,:));


[edgeOriginIdx, edgeDestIdx] = ndgrid(supplyNodes, demandNodes);
edgeOriginIdx = edgeOriginIdx(:);          
edgeDestIdx   = edgeDestIdx(:);            
nEdges = numel(edgeOriginIdx);              


edgeId = reshape(1:nEdges, [numel(supplyNodes), numel(demandNodes)]);


scenarioNames = ["A_single_hub", "B_demand_shift", "C_cost_shock", "D_dual_hub", "E_permanent_shift"];
nSc = numel(scenarioNames);


C_threshold_km = 60;
C_factor       = 1.8;
C_active_times = 1:3;

B_inc_node13 = 1.4;     
B_dec_node14 = 0.7;     
B_decay      = 0.3;     


A_node = 4; A_start = 0.4; A_end = 1.0;


D_node3 = 3; D3_start = 0.5; D3_end = 1.0;
D_node4 = 4; D4_start = 0.4; D4_end = 1.0;


E_node = 4; E_start = 0.4; E_plateau = 0.8; E_plateau_time = 6;


F = cell(nSc, T+1);        
M = zeros(nSc, T+1);       
mu = cell(nSc, T+1);         
dEff = cell(nSc, T+1);      

[dEff0, S_base0, D_base0] = scenario_params( ...
    "BASE", 0, S0, D0, dBase, ...
    A_node, A_start, A_end, ...
    B_inc_node13, B_dec_node14, B_decay, ...
    C_threshold_km, C_factor, C_active_times, ...
    D_node3, D3_start, D3_end, D_node4, D4_start, D4_end, ...
    E_node, E_start, E_plateau, E_plateau_time);

F0 = gravity_ipf_flow(S_base0, D_base0, dEff0); % 5x5, row sums S, col sums D (balanced)
M0 = sum(F0,'all');
mu0 = reshape(F0, [], 1) / M0;


for sc = 1:nSc
    F{sc,1} = F0; M(sc,1) = M0; mu{sc,1} = mu0; dEff{sc,1} = dEff0;
    for t = 1:T
        [dE, S_t, D_t] = scenario_params( ...
            scenarioNames(sc), t, S0, D0, dBase, ...
            A_node, A_start, A_end, ...
            B_inc_node13, B_dec_node14, B_decay, ...
            C_threshold_km, C_factor, C_active_times, ...
            D_node3, D3_start, D3_end, D_node4, D4_start, D4_end, ...
            E_node, E_start, E_plateau, E_plateau_time);

       
        if abs(sum(S_t) - sum(D_t)) > 1e-10
            D_t = D_t * (sum(S_t)/sum(D_t));
        end

        Ft = gravity_ipf_flow(S_t, D_t, dE);
        Mt = sum(Ft,'all');
        mut = reshape(Ft, [], 1) / Mt;

        F{sc,t+1} = Ft;
        M(sc,t+1) = Mt;
        mu{sc,t+1} = mut;
        dEff{sc,t+1} = dE;
    end
end

C_edge = zeros(nEdges, nEdges);
for e1 = 1:nEdges
    i = edgeOriginIdx(e1); j = edgeDestIdx(e1);
    for e2 = 1:nEdges
        k = edgeOriginIdx(e2); l = edgeDestIdx(e2);
        C_edge(e1,e2) = norm(XY(i,:) - XY(k,:), 2) + norm(XY(j,:) - XY(l,:), 2);
    end
end


W1 = zeros(nSc, T+1);
Rs = zeros(nSc, T+1);


lpOpts = optimoptions('linprog', ...
    'Algorithm','dual-simplex', ...
    'Display','none', ...
    'OptimalityTolerance',1e-9, ...
    'ConstraintTolerance',1e-9);


for sc = 1:nSc
    for tt = 1:(T+1)
        W1(sc,tt) = wasserstein_LP(mu0, mu{sc,tt}, C_edge, lpOpts);
    end
end

Wmax = max(W1(:));
if Wmax < 1e-12
    warning("Wmax ~ 0; check flow generation.");
    Wmax = 1;
end
Rs = 1 - (W1 / Wmax);


TP = M / M0;

dbar = zeros(nSc, T+1);
Cagg = zeros(nSc, T+1);
CI = zeros(nSc, T+1);


C0 = sum(F0 .* dEff0, 'all');

for sc = 1:nSc
    for tt = 1:(T+1)
        Ft  = F{sc,tt};
        mut = mu{sc,tt};
        dE  = dEff{sc,tt};

        
        dbar(sc,tt) = sum(reshape(mut, size(dE)) .* dE, 'all');

       
        Cagg(sc,tt) = sum(Ft .* dE, 'all');
    end
end
CI = Cagg / C0;


timeVec = 0:T;


minRs = min(Rs, [], 2);
tMinRs = zeros(nSc,1);
recThr = 0.90;
tRec = nan(nSc,1);

for sc = 1:nSc
    [~, idx] = min(Rs(sc,:));
    tMinRs(sc) = timeVec(idx);

    idxRec = find(Rs(sc,:) >= recThr, 1, 'first');
    if ~isempty(idxRec)
        tRec(sc) = timeVec(idxRec);
    end
end


structLoss = sum(1 - Rs, 2);


peakCI = max(CI, [], 2);
peakDbar = max(dbar, [], 2);

summaryTbl = table( ...
    scenarioNames(:), minRs, tMinRs, tRec, structLoss, peakCI, peakDbar, ...
    'VariableNames', {'Scenario','Min_Rs','t_at_Min_Rs','t_Recovery_Rs_ge_0p90','StructuralLoss_sum1mRs','Peak_CI','Peak_MeanDistance'} );

writetable(summaryTbl, fullfile(TAB_DIR, "Table_summary_scenarios.csv"));


rows = nSc * (T+1);
ScenarioCol = strings(rows,1);
tCol = zeros(rows,1);
W1Col = zeros(rows,1);
RsCol = zeros(rows,1);
TPCol = zeros(rows,1);
dbarCol = zeros(rows,1);
CICol = zeros(rows,1);

r = 0;
for sc = 1:nSc
    for tt = 1:(T+1)
        r = r + 1;
        ScenarioCol(r) = scenarioNames(sc);
        tCol(r) = timeVec(tt);
        W1Col(r) = W1(sc,tt);
        RsCol(r) = Rs(sc,tt);
        TPCol(r) = TP(sc,tt);
        dbarCol(r) = dbar(sc,tt);
        CICol(r) = CI(sc,tt);
    end
end

longTbl = table(ScenarioCol, tCol, W1Col, RsCol, TPCol, dbarCol, CICol, ...
    'VariableNames', {'Scenario','t','W1','Rs','TP','MeanDistance','CI'});
writetable(longTbl, fullfile(TAB_DIR, "Table_metrics_long.csv"));


fig = figure('Name','Rs(t)'); hold on; grid on;
for sc = 1:nSc
    plot(timeVec, Rs(sc,:), 'LineWidth', 1.8);
end
xlabel('t'); ylabel('R_s(t)');
title('Structural Resilience Index R_s(t) across scenarios');
legend(scenarioNames, 'Interpreter','none','Location','best');
saveas(fig, fullfile(FIG_DIR, "Fig_Rs_curves.png"));


fig = figure('Name','W1(t)'); hold on; grid on;
for sc = 1:nSc
    plot(timeVec, W1(sc,:), 'LineWidth', 1.8);
end
xlabel('t'); ylabel('W_1(\mu^0,\mu^t)');
title('Wasserstein structural displacement W_1 across scenarios');
legend(scenarioNames, 'Interpreter','none','Location','best');
saveas(fig, fullfile(FIG_DIR, "Fig_W1_curves.png"));


fig = figure('Name','TP(t)'); hold on; grid on;
for sc = 1:nSc
    plot(timeVec, TP(sc,:), 'LineWidth', 1.8);
end
xlabel('t'); ylabel('TP(t)=M(t)/M(0)');
title('Throughput ratio TP(t) across scenarios');
legend(scenarioNames, 'Interpreter','none','Location','best');
saveas(fig, fullfile(FIG_DIR, "Fig_TP_curves.png"));


fig = figure('Name','MeanDistance'); hold on; grid on;
for sc = 1:nSc
    plot(timeVec, dbar(sc,:), 'LineWidth', 1.8);
end
xlabel('t'); ylabel('\bar d(t)');
title('Mean shipment distance \bar d(t) across scenarios');
legend(scenarioNames, 'Interpreter','none','Location','best');
saveas(fig, fullfile(FIG_DIR, "Fig_MeanDistance_curves.png"));


fig = figure('Name','CI(t)'); hold on; grid on;
for sc = 1:nSc
    plot(timeVec, CI(sc,:), 'LineWidth', 1.8);
end
xlabel('t'); ylabel('CI(t)=C(t)/C(0)');
title('Transport cost index CI(t) across scenarios');
legend(scenarioNames, 'Interpreter','none','Location','best');
saveas(fig, fullfile(FIG_DIR, "Fig_CI_curves.png"));


fig = figure('Name','Rs_vs_TP'); hold on; grid on;
for sc = 1:nSc
    scatter(Rs(sc,:), TP(sc,:), 40, 'filled');
end
xlabel('R_s'); ylabel('TP');
title('Structural vs performance: R_s vs TP (all times)');
legend(scenarioNames, 'Interpreter','none','Location','best');
saveas(fig, fullfile(FIG_DIR, "Fig_Rs_vs_TP_scatter.png"));


fig = figure('Name','Rs_vs_CI'); hold on; grid on;
for sc = 1:nSc
    scatter(Rs(sc,:), CI(sc,:), 40, 'filled');
end
xlabel('R_s'); ylabel('CI');
title('Structural vs performance: R_s vs CI (all times)');
legend(scenarioNames, 'Interpreter','none','Location','best');
saveas(fig, fullfile(FIG_DIR, "Fig_Rs_vs_CI_scatter.png"));


plot_flow_heatmap(F0, "Baseline flows F(0)", fullfile(FIG_DIR,"Fig_heatmap_baseline_F0.png"));


scA = find(scenarioNames=="A_single_hub");
scC = find(scenarioNames=="C_cost_shock");
scE = find(scenarioNames=="E_permanent_shift");

plot_flow_heatmap(F{scA,2}, "Scenario A flows at t=1", fullfile(FIG_DIR,"Fig_heatmap_A_t1.png"));
plot_flow_heatmap(F{scC,3}, "Scenario C flows at t=2", fullfile(FIG_DIR,"Fig_heatmap_C_t2.png"));
plot_flow_heatmap(F{scE,end}, "Scenario E flows at t=10", fullfile(FIG_DIR,"Fig_heatmap_E_t10.png"));


plot_network_graph(XY, nNodes, edgeOriginIdx, edgeDestIdx, F0, ...
    "Network graph: Baseline t=0", fullfile(FIG_DIR,"Fig_graph_baseline_t0.png"));


plot_network_graph(XY, nNodes, edgeOriginIdx, edgeDestIdx, F{scA,2}, ...
    "Network graph: Scenario A, t=1", fullfile(FIG_DIR,"Fig_graph_A_t1.png"));


plot_network_graph(XY, nNodes, edgeOriginIdx, edgeDestIdx, F{scC,3}, ...
    "Network graph: Scenario C, t=2", fullfile(FIG_DIR,"Fig_graph_C_t2.png"));


plot_network_graph(XY, nNodes, edgeOriginIdx, edgeDestIdx, F{scE,end}, ...
    "Network graph: Scenario E, t=10", fullfile(FIG_DIR,"Fig_graph_E_t10.png"));


K = 8;
topTblAll = table();
for sc = 1:nSc
    [~, idxMaxW] = max(W1(sc,:));
    mut = mu{sc, idxMaxW};
    delta = abs(mut - mu0);

    [sortedDelta, idxEdges] = sort(delta, 'descend');
    idxEdges = idxEdges(1:K);
    sortedDelta = sortedDelta(1:K);

    origins = edgeOriginIdx(idxEdges);
    dests   = edgeDestIdx(idxEdges);

    tmp = table( ...
        repmat(scenarioNames(sc), K, 1), ...
        repmat(timeVec(idxMaxW), K, 1), ...
        idxEdges, origins, dests, sortedDelta, ...
        'VariableNames', {'Scenario','t_at_maxW1','EdgeID','OriginNode','DemandNode','abs_mu_change'} );
    topTblAll = [topTblAll; tmp]; 
end
writetable(topTblAll, fullfile(TAB_DIR, "Table_top_OD_changes_at_maxW1.csv"));

disp("Done. Outputs written to: " + OUT_DIR);



function ensure_dir(d)
    if ~exist(d, 'dir'); mkdir(d); end
end

function D = pairwise_dist(A, B)
   
    m = size(A,1); n = size(B,1);
    D = zeros(m,n);
    for i=1:m
        for j=1:n
            D(i,j) = norm(A(i,:) - B(j,:), 2);
        end
    end
end

function [dE, S, D] = scenario_params( ...
    scName, t, S0, D0, dBase, ...
    A_node, A_start, A_end, ...
    B_inc13, B_dec14, B_decay, ...
    C_thr, C_factor, C_times, ...
    D_node3, D3_start, D3_end, D_node4, D4_start, D4_end, ...
    E_node, E_start, E_plateau, E_plateau_time)

    S = S0; D = D0; dE = dBase;

    if scName == "BASE" || t==0
        return;
    end

    switch scName
        case "A_single_hub"
           
            f = lin_factor(t, 1, 10, A_start, A_end);
            S(A_node) = S0(A_node) * f;

        case "B_demand_shift"
           
            D1 = D0;
           
            D1(3) = B_inc13 * D0(3);
            D1(4) = B_dec14 * D0(4);

           
            D = D0 + (D1 - D0) * exp(-B_decay * t);

        case "C_cost_shock"
            
            if ismember(t, C_times)
                dE = dBase;
                mask = dBase > C_thr;
                dE(mask) = C_factor * dBase(mask);
            end

        case "D_dual_hub"
            f3 = lin_factor(t, 1, 10, D3_start, D3_end);
            f4 = lin_factor(t, 1, 10, D4_start, D4_end);
            S(D_node3) = S0(D_node3) * f3;
            S(D_node4) = S0(D_node4) * f4;

        case "E_permanent_shift"
            
            if t <= E_plateau_time
                f = lin_factor(t, 1, E_plateau_time, E_start, E_plateau);
            else
                f = E_plateau;
            end
            S(E_node) = S0(E_node) * f;

        otherwise
            error("Unknown scenario: " + scName);
    end
end

function f = lin_factor(t, t0, t1, f0, f1)
    
    if t <= t0
        f = f0;
    elseif t >= t1
        f = f1;
    else
        f = f0 + (f1 - f0) * ((t - t0) / (t1 - t0));
    end
end

function F = gravity_ipf_flow(S, D, distMat)
    

    epsd = 1e-9;
    W = 1 ./ max(distMat, epsd);  

   
    F = W;
    
    maxIter = 2000;
    tol = 1e-10;

    for it = 1:maxIter
       
        rs = sum(F,2);
        rowScale = S ./ max(rs, epsd);
        F = F .* rowScale;

        
        cs = sum(F,1);
        colScale = (D' ./ max(cs, epsd));
        F = F .* colScale;

        
        if max(abs(sum(F,2)-S)) < tol && max(abs(sum(F,1)'-D)) < tol
            break;
        end
    end
end

function W = wasserstein_LP(mu0, mu1, C, lpOpts)
    

    mu0 = mu0(:); mu1 = mu1(:);
    n = numel(mu0);
    if numel(mu1) ~= n
        error("mu0 and mu1 must have same length.");
    end

    c = C(:); 

    
    Arow = kron(speye(n), ones(1,n));     
    brow = mu0;

    
    Acol = kron(ones(1,n), speye(n));    
    bcol = mu1;

    Aeq = [Arow; Acol];
    beq = [brow; bcol];

    lb = zeros(n*n,1);
    ub = []; 

    x = linprog(c, [], [], Aeq, beq, lb, [], lpOpts);
    if isempty(x)
        error("linprog failed to find a solution (check feasibility).");
    end

    W = c' * x;
end

function plot_flow_heatmap(F, ttl, outPath)
    fig = figure('Name', ttl);
    imagesc(F);
    axis tight; colorbar;
    xlabel('Demand nodes (11..15 as columns 1..5)');
    ylabel('Supply nodes (1..5)');
    title(ttl);
    saveas(fig, outPath);
end

function plot_network_graph(XY, nNodes, edgeOriginIdx, edgeDestIdx, F5x5, ttl, outPath)
    
    W = reshape(F5x5, [], 1);
    G = digraph(edgeOriginIdx, edgeDestIdx, W, nNodes);

    fig = figure('Name', ttl);
    p = plot(G, ...
        'XData', XY(:,1), 'YData', XY(:,2), ...
        'NodeLabel', arrayfun(@num2str, 1:nNodes, 'UniformOutput', false), ...
        'MarkerSize', 6);

    title(ttl);
    grid on;
    axis equal;

    
    w = G.Edges.Weight;
    if all(w==0)
        lw = ones(size(w));
    else
        lw = 0.5 + 5.0 * (w / max(w)); 
    end
    p.LineWidth = lw;

  

    saveas(fig, outPath);
end