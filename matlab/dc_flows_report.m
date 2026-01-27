function dc_flows_report(case_name, plan_path, xi_path, limits_path, out_path)
% DC_FLOWS_REPORT
% Solve the same DC verifier as verify_plan/run_ground_truth, but also output:
%  - line flows P
%  - limits Pmax
%  - loading ratios |P|/Pmax for active (gate=1) lines
%  - counts near limits (>=0.95, >=0.99) and top loaded lines

clc; yalmip('clear');

%% --- Load case ---
switch lower(case_name)
    case 'case118'
        mpopt = mpoption('verbose', 0, 'out.all', 0);
        mpc   = rundcpf('case118', mpopt);
    otherwise
        error('Unsupported case: %s', case_name);
end
nb = size(mpc.bus,1);
nl = size(mpc.branch,1);

%% --- Inputs ---
limits  = read_limits_yaml(limits_path);
plan    = jsondecode(fileread(plan_path));
xi_raw  = jsondecode(fileread(xi_path));
xi      = normalize_xi(xi_raw, nl);

corrmap = jsondecode(fileread('config/corridor_map.json'));

%% --- Build (z, delta_g) from plan (same logic as verify_plan) ---
z = ones(nl,1);
z(xi==0) = 0;  % PSPS forced-open

budget       = value_or_default(limits,'budget',3);
toggles_used = 0;

ca_list = normalize_actions(plan, 'corridor_actions');

for a = 1:numel(ca_list)
    if toggles_used >= budget, break; end
    act = to_str_scalar(ca_list(a), 'action');
    if act ~= "open", continue; end

    lid = get_numeric_field(ca_list(a), 'line', NaN);
    if isfinite(lid)
        lid = round(lid);
        if lid>=1 && lid<=nl && xi(lid)==1 && z(lid)==1
            z(lid) = 0; toggles_used = toggles_used + 1;
            continue;
        end
    end

    S = to_str_scalar(ca_list(a), 'name');
    if S ~= "" && isfield(corrmap, S)
        ids = corrmap.(S); ids = ids(:).';
        pick = [];
        for id = ids
            id = round(id);
            if id>=1 && id<=nl && xi(id)==1 && z(id)==1
                pick = id; break;
            end
        end
        if ~isempty(pick)
            z(pick) = 0; toggles_used = toggles_used + 1;
        end
    end
end

delta_g = zeros(nb,1); % no redispatch here
gate = (xi(:).*z(:));

%% --- Solve DC LP and extract flows ---
try
    [feasible,J,shedMW,Psol,Pmax] = run_dc_with_flows(mpc, gate, delta_g, limits);
catch ME
    warning("dc_flows_report error: %s", ME.message);
    feasible=false; J=1e9; shedMW=NaN;
    Psol=nan(nl,1); Pmax=nan(nl,1);
end

%% --- Summaries: loading & near-binding counts ---
tol95 = 0.95;
tol99 = 0.99;

loading = abs(Psol) ./ (Pmax + 1e-12);
active  = (gate > 0.5) & isfinite(loading);

max_loading = NaN;
n95 = 0; n99 = 0;
top_lines = struct('line',{},'flow',{},'limit',{},'ratio',{});

if any(active)
    la = loading(active);
    max_loading = max(la);
    n95 = sum(la >= tol95);
    n99 = sum(la >= tol99);

    idx_active = find(active);
    [~,ord] = sort(loading(idx_active), 'descend');
    K = min(20, numel(ord));
    for k=1:K
        j = idx_active(ord(k));
        top_lines(end+1).line  = j; %#ok<AGROW>
        top_lines(end).flow   = Psol(j);
        top_lines(end).limit  = Pmax(j);
        top_lines(end).ratio  = loading(j);
    end
end

%% --- Write JSON ---
out = struct();
out.feasible     = logical(feasible);
out.J            = J;
out.shed_MW      = shedMW;
out.max_loading  = max_loading;
out.n_ge_95      = n95;
out.n_ge_99      = n99;
out.top_lines    = top_lines;

fid=fopen(out_path,'w');
fprintf(fid,'%s', jsonencode(out));
fclose(fid);

end

% ================= Helpers =================

function ca = normalize_actions(plan, fieldname)
ca = struct([]);
if ~isstruct(plan) || ~isfield(plan, fieldname) || isempty(plan.(fieldname))
    return;
end
v = plan.(fieldname);
if isstruct(v), ca = v; else, ca = struct([]); end
end

function xi = normalize_xi(xi_raw, n_line)
xi = ones(n_line,1);
if isstruct(xi_raw)
    if isfield(xi_raw,'xi')
        v = double(xi_raw.xi(:) ~= 0);
        xi = fit_mask_len(v, n_line);
    elseif isfield(xi_raw,'forced_open')
        idx = round(xi_raw.forced_open(:));
        idx = idx(idx>=1 & idx<=n_line);
        xi = ones(n_line,1); xi(idx)=0;
    else
        error('xi struct must include "xi" or "forced_open".');
    end
elseif isnumeric(xi_raw) || islogical(xi_raw)
    v = xi_raw(:);
    u = unique(v(~isnan(v)));
    if all(ismember(u',[0 1]))
        v  = double(v~=0);
        xi = fit_mask_len(v, n_line);
    else
        v = round(v); v = v(v>=1 & v<=n_line);
        xi = ones(n_line,1); xi(v)=0;
    end
else
    error('Unsupported xi JSON format.');
end
end

function xi = fit_mask_len(v, n_line)
if numel(v) < n_line
    xi = ones(n_line,1);
    xi(1:numel(v)) = v;
elseif numel(v) > n_line
    xi = v(1:n_line);
else
    xi = v;
end
end

function L = read_limits_yaml(path)
txt = fileread(path);
L = struct();
L.budget         = str2double_maybe(extract_first(txt, 'budget:\s*([0-9\.\-eE]+)'));
L.gamma          = str2double_maybe(extract_first(txt, 'gamma:\s*([0-9\.\-eE]+)'));
L.lambda         = str2double_maybe(extract_first(txt, 'lambda:\s*([0-9\.\-eE]+)'));
if isnan(L.gamma),  L.gamma  = 100; end
if isnan(L.lambda), L.lambda = 0;   end
end

function val = extract_first(txt, pat)
m = regexp(txt, pat, 'tokens', 'once');
if isempty(m), val = ''; else, val = m{1}; end
end

function x = str2double_maybe(s)
if isempty(s), x = NaN; else, x = str2double(s); end
end

function out = value_or_default(S, key, def)
if isfield(S, key) && ~isempty(S.(key)) && all(isfinite(S.(key)))
    out = S.(key);
else
    out = def;
end
end

function s = to_str_scalar(st, field)
if ~isfield(st, field) || isempty(st.(field)), s = ""; return; end
v = st.(field);
if isstring(v), s = v(1); return; end
if ischar(v),   s = string(v); return; end
s = string(v);
end

function v = get_numeric_field(st, field, def)
if ~isfield(st, field) || isempty(st.(field)), v = def; return; end
v = st.(field);
if ~isnumeric(v) || ~isscalar(v) || ~isfinite(v), v = def; end
end

% ===== DC LP (same as verify_plan/run_ground_truth), but return P and Pmax =====
function [feasible,J,shedMW,Psol,Pmax] = run_dc_with_flows(mpc, gate, delta_g, limits)
yalmip('clear');
options = sdpsettings('verbose', 0);

nb = size(mpc.bus,1);
nl = size(mpc.branch,1);

% costs
c = zeros(nb,1);
c(10)=0.217; c(12)=1.052; c(25)=0.434; c(26)=0.308; c(31)=5.882; c(46)=3.448;
c(49)=0.467; c(54)=1.724; c(59)=0.606; c(61)=0.588; c(65)=0.2493; c(66)=0.2487;
c(69)=0.1897; c(80)=0.205; c(87)=7.142; c(92)=10; c(100)=0.381; c(103)=2; c(111)=2.173;

Pd = mpc.bus(:,3); Pd(90)=440;

Pmax = 220*ones(nl,1);
b440 = [3 21 31 33 50 96 98 99 90 93 94 97 107 108 116 123 137 163];
b660 = [38 36 51 138 140];
Pmax(b440)=440; Pmax(b660)=660; Pmax(7)=1100; Pmax(9)=1100; Pmax(8)=880;

Bline = 1./mpc.branch(:,4);
A = zeros(nb,nl);
for i=1:nb
    for j=1:nl
        if mpc.branch(j,1)==i, A(i,j)= 1; end
        if mpc.branch(j,2)==i, A(i,j)=-1; end
    end
end
M = zeros(nl,nb);
for j=1:nl
    i1=mpc.branch(j,1); i2=mpc.branch(j,2);
    M(j,i1)= Bline(j); M(j,i2)=-Bline(j);
end

Png_min = zeros(nb,1);
Png_max = zeros(nb,1);
Png_max(10)=550; Png_max(12)=185; Png_max(25)=320; Png_max(26)=414; Png_max(31)=107;
Png_max(46)=119; Png_max(49)=304; Png_max(54)=148; Png_max(59)=255; Png_max(61)=260;
Png_max(65)=491; Png_max(66)=492; Png_max(69)=805.2; Png_max(80)=577; Png_max(87)=104;
Png_max(92)=100; Png_max(100)=352; Png_max(103)=140; Png_max(111)=136;

theta = sdpvar(nb,1);
P     = sdpvar(nl,1);
Ps    = sdpvar(nb,1);
Png   = sdpvar(nb,1);

gamma  = value_or_default(limits,'gamma',100);
lambda = value_or_default(limits,'lambda',0);

obj = c'*Png + gamma*sum(Ps) + lambda*sum(abs(delta_g));

con = {};
con{end+1} = -ones(nb,1) <= theta <= ones(nb,1);
con{end+1} = theta(1) == 0;
con{end+1} = Png_min <= Png + delta_g <= Png_max;
con{end+1} = A*P == (Png + delta_g) - (Pd - Ps);
con{end+1} = Ps >= 0;

con{end+1} = -Pmax.*gate <= P <= Pmax.*gate;

Moff=1000; scale=100;
for j=1:nl
    con{end+1} =  scale*M(j,:)*theta - P(j) + (1-gate(j))*Moff >= 0;
    con{end+1} =  scale*M(j,:)*theta - P(j) - (1-gate(j))*Moff <= 0;
end

sol = optimize([con{:}], obj, options);
feasible = (sol.problem==0);

if feasible
    J = value(obj);
    shedMW = sum(value(Ps));
    Psol = value(P);
else
    J = 1e9; shedMW = NaN;
    Psol = nan(nl,1);
end
end
