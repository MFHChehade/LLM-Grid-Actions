function make_summary(case_name, xi_path, limits_path, out_path)
% MAKE_SUMMARY
% Reads PSPS mask (flexible formats), case, and limits; emits a compact
% JSON summary used by the inference/planning loop.
%
% Output JSON fields:
%   {
%     "case": "case118",
%     "buses": 118,
%     "lines": 186,
%     "psps_forced_open_count": <int>,
%     "impacted_corridors": {
%        "S1": {
%           "lines": [..],
%           "forced_open_lines": [..],           % subset where xi==0
%           "eligible_to_open_lines": [..]       % xi==1
%        },
%        "S2": {...}
%     },
%     "toggle_budget": <from limits>,
%     "redispatch_cap": <from limits>,
%     "allowed_actions": [
%       "open(corridor_name[, line_id])",
%       "close(corridor_name[, line_id])",
%       "nudge(bus, up|down)"
%     ]
%   }

clc;

%% --- Load case ---
switch lower(case_name)
    case 'case118'
        mpc = rundcpf('case118');
    otherwise
        error('Unsupported case: %s', case_name);
end
n_bus  = size(mpc.bus,1);
n_line = size(mpc.branch,1);

%% --- Load inputs ---
limits = read_limits_yaml(limits_path);

% Get the directory where limits.yml lives
[config_dir, ~, ~] = fileparts(limits_path);
corrmap_path = fullfile(config_dir, 'corridor_map.json');

corrmap = jsondecode(fileread(corrmap_path));


% xi normalization (robust to struct/mask/index list)
xi_raw = jsondecode(fileread(xi_path));
xi = normalize_xi(xi_raw, n_line);

%% --- Build corridor-based view ---
impacted = struct();
cm_names = fieldnames(corrmap);
for k = 1:numel(cm_names)
    name = cm_names{k};
    ids = corrmap.(name)(:);
    ids = ids(ids>=1 & ids<=n_line);

    forced_open = ids(xi(ids)==0);
    elig_open   = ids(xi(ids)==1);

    % mark only corridors that either intersect PSPS or are defined
    entry = struct();
    entry.lines                = ids(:).';
    entry.forced_open_lines    = forced_open(:).';
    entry.eligible_to_open_lines = elig_open(:).';
    impacted.(name) = entry;
end

%% --- Compose summary ---
S = struct();
S.case   = string(case_name);
S.buses  = n_bus;
S.lines  = n_line;
S.psps_forced_open_count = sum(xi==0);
S.impacted_corridors = impacted;
S.toggle_budget  = value_or_default(limits,'budget',3);
S.redispatch_cap = value_or_default(limits,'redispatch_cap',5);
S.allowed_actions = [ ...
    "open(corridor_name[, line_id])", ...
    "close(corridor_name[, line_id])", ...
    "nudge(bus, up|down)" ...
];

% Write
fid = fopen(out_path,'w'); assert(fid>0,'Cannot open output: %s', out_path);
fprintf(fid,'%s', jsonencode(S));
fclose(fid);
end

% ===== Utilities (duplicated here so this file is standalone) =====
function xi = normalize_xi(xi_raw, n_line)
xi = ones(n_line,1);
if isstruct(xi_raw)
    if isfield(xi_raw,'xi')
        v = double(xi_raw.xi(:) ~= 0);
    elseif isfield(xi_raw,'forced_open')
        v = ones(n_line,1);
        idx = round(xi_raw.forced_open(:));
        idx = idx(idx>=1 & idx<=n_line);
        v(idx) = 0;
        xi = v; return
    else
        error('xi struct must contain "xi" or "forced_open".');
    end
elseif isnumeric(xi_raw) || islogical(xi_raw)
    v = xi_raw(:);
    u = unique(v(~isnan(v)));
    if all(ismember(u',[0 1]))
        v = double(v ~= 0);
    elseif all(v>=1 & v<=n_line) && numel(v)<n_line
        v = ones(n_line,1); v(round(xi_raw(:))) = 0; xi=v; return
    else
        error('Unsupported xi numeric format.');
    end
else
    error('Unsupported xi JSON.');
end

if numel(v) < n_line
    xi(1:numel(v)) = v; xi(numel(v)+1:end) = 1;
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
L.redispatch_cap = str2double_maybe(extract_first(txt, 'redispatch_cap:\s*([0-9\.\-eE]+)'));
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
if isfield(S, key) && ~isempty(S.(key)) && isfinite(S.(key))
    out = S.(key);
else
    out = def;
end
end
