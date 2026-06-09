function report = retrain_air_quality_model_V3(indicator, trainStart, trainEnd, ...
    testStart, testEnd, userOpts)
%RETRAIN_AIR_QUALITY_MODEL_V3 日模型用日数据，月模型用月数据，支持 PM2.5/O3 新增列
%
%   PM2.5 新模型增加 O 列（原模型不含 O）
%   O3    新模型增加 N 列（原模型不含 N）
%   其他污染物无新增列
%
%   输入：
%     indicator  : 'PM2.5' | 'PM10' | 'O3' | 'NO2' | 'CO' | 'SO2'
%     trainStart/trainEnd : 训练集日期区间 (字符串 'yyyy-mm-dd' 或 datenum)
%     testStart/testEnd   : 测试集日期区间 (同上)
%     userOpts   : (可选) 结构体，可含字段 modelTypes: 'both'(默认)|'daily'|'monthly'
%                  也可以直接传入字符串 'daily' 或 'monthly'，自动转换。
%
%   说明：
%     日模型（daily）从“日均值预测”表读取，按日期区间划分。
%     月度模型（monthly）直接从“月均值预测3”表中读取历史数据，
%     按日期区间划分训练/测试集，不再依赖日数据滑动窗口。

if nargin < 5
    error('retrain_air_quality_model:NotEnoughInputs', ...
        '需要 indicator 与训练/测试日期区间。');
end
% ============ 自适应处理 userOpts 为字符串或结构体 ============
if nargin < 6 || isempty(userOpts)
    userOpts = struct();
elseif ischar(userOpts) || (isstring(userOpts) && isscalar(userOpts))
    userOpts = struct('modelTypes', char(userOpts));
elseif ~isstruct(userOpts)
    error('retrain_air_quality_model:BadUserOpts', ...
        'userOpts 必须是结构体、字符串或空值。');
end
% ====================================================================

rootDir = scriptRootDir();
addpath(rootDir);
addpath(fullfile(rootDir, 'addpath'));
try
    pkg load io; %#ok<*NASGU>
catch
end

opts = defaultOptions(rootDir);
opts = mergeOptions(opts, userOpts);

% ---- 解析要训练的模型类型 ----
if ~isfield(opts, 'modelTypes') || isempty(opts.modelTypes)
    opts.modelTypes = 'both';
end
validTypes = {'both','daily','monthly'};
if ~any(strcmpi(opts.modelTypes, validTypes))
    error('retrain_air_quality_model:InvalidModelTypes', ...
        'modelTypes 必须是 ''both'', ''daily'' 或 ''monthly''。');
end
trainDaily = any(strcmpi(opts.modelTypes, {'both','daily'}));
trainMonthly = any(strcmpi(opts.modelTypes, {'both','monthly'}));

[cfg, tag] = indicatorConfig(indicator);
if isfield(opts, 'workbook') && ~isempty(opts.workbook)
    cfg.workbook = opts.workbook;
end
predictFile = fullfile(rootDir, sprintf('predict_fixed_air_quality_%s.m', tag));
if ~exist(predictFile, 'file')
    error('retrain_air_quality_model:PredictFileMissing', ...
        '找不到原预测文件：%s', predictFile);
end

trainStartNum = parseDateValue(trainStart);
trainEndNum   = parseDateValue(trainEnd);
testStartNum  = parseDateValue(testStart);
testEndNum    = parseDateValue(testEnd);
if trainEndNum < trainStartNum || testEndNum < testStartNum
    error('retrain_air_quality_model:InvalidRange', '结束日期不能早于起始日期。');
end

% ========== 初始化 report 结构，确保所有字段都存在 ==========
report = struct();
report.indicator = upper(indicator);
report.tag = tag;
report.trainStart = datestr(trainStartNum, 'yyyy-mm-dd');
report.trainEnd   = datestr(trainEndNum, 'yyyy-mm-dd');
report.testStart  = datestr(testStartNum, 'yyyy-mm-dd');
report.testEnd    = datestr(testEndNum, 'yyyy-mm-dd');
report.messages = {};
% 日模型相关字段默认值
report.dailyTrainN = [];
report.dailyTestN = [];
report.daily = struct('trained', false, 'improved', false, ...
    'orig', struct(), 'new', struct(), 'series', []);
% 月度模型相关字段默认值
report.monthly = struct('trained', false, 'available', false, 'improved', false, ...
    'orig', struct(), 'new', struct(), 'series', []);
report.candidateFile = '';
report.updated = false;
report.outputDir = '';
report.summaryFile = '';
% 其它可能在后面赋值的字段
report.newPredictorCols = cfg.predictorCols;
report.originalDailyPredictorCols = cfg.origPredictorCols;
report.originalMonthlyPredictorCols = cfg.origPredictorCols;

fprintf('=== 重训练 %s ===\n', report.indicator);
fprintf('训练集 %s ~ %s，测试集 %s ~ %s\n', report.trainStart, report.trainEnd, ...
    report.testStart, report.testEnd);
fprintf('训练模型类型：%s\n', opts.modelTypes);
fprintf('新模型特征列：%s\n', strjoin(cfg.predictorCols, ', '));
fprintf('原模型特征列：%s\n', strjoin(cfg.origPredictorCols, ', '));

% ---------------- 读取原模型结构 ----------------
origDaily = embeddedModelStruct(predictFile, 'daily');
origMonthly = embeddedModelStruct(predictFile, 'monthly');

% ===================== 日尺度数据准备 =====================
if trainDaily
    [P_all, T_all, D_all] = readTargetBlock(cfg, opts.inputDir, ...
        cfg.dailyStartRow, cfg.dailyEndRow, true);
    [P_all, T_all, D_all] = keepCompleteSamples(P_all, T_all, D_all);
    [P_all, T_all, D_all] = sortSamplesByDate(P_all, T_all, D_all);

    trainMask = D_all >= trainStartNum & D_all <= trainEndNum;
    testMask  = D_all >= testStartNum  & D_all <= testEndNum;

    P_train = P_all(:, trainMask); T_train = T_all(trainMask); D_train = D_all(trainMask);
    P_test  = P_all(:, testMask);  T_test  = T_all(testMask);  D_test  = D_all(testMask);

    if numel(T_train) < 10
        error('retrain_air_quality_model:TooFewTrain', ...
            '训练集样本过少（%d 条），请扩大训练日期区间。', numel(T_train));
    end
    if numel(T_test) < 1
        error('retrain_air_quality_model:NoTest', ...
            '测试日期区间内没有可用样本，请重新选择。');
    end
    report.dailyTrainN = numel(T_train);
    report.dailyTestN = numel(T_test);
    
    fprintf('训练 daily 模型 …\n');
    newDaily = trainOneModel(P_train, T_train, opts, cfg, 'daily');
    newDailyPredTest = predictWithModel(newDaily, P_test);
    report.daily.new = calcMetrics(T_test, newDailyPredTest);

    % 原 daily 嵌入模型：对于 PM2.5 和 O3，原模型特征比新模型少一列
    if strcmpi(indicator, 'PM2.5') || strcmpi(indicator, 'O3')
        if size(P_test,1) < 2
            error('特征行数不足，无法裁剪最后一列。当前特征数：%d', size(P_test,1));
        end
        P_test_orig = P_test(1:end-1, :);
        fprintf('原日模型使用前 %d 个特征（共 %d），已裁剪最后一列。\n', size(P_test_orig,1), size(P_test,1));
    else
        P_test_orig = P_test;
        fprintf('原日模型使用全部 %d 个特征。\n', size(P_test,1));
    end
    origDailyPredTest = predictWithModel(origDaily, P_test_orig);
    report.daily.orig = calcMetrics(T_test, origDailyPredTest);

    report.daily.series = struct('dates', D_test, 'actual', T_test, ...
        'newPred', newDailyPredTest, 'origPred', origDailyPredTest);
    report.daily.improved = isBetter(report.daily.new, report.daily.orig);
    fprintf('  daily  新 RMSE=%.4g R2=%.4g | 原 RMSE=%.4g R2=%.4g | 更优=%d\n', ...
        report.daily.new.rmse, report.daily.new.R2, ...
        report.daily.orig.rmse, report.daily.orig.R2, report.daily.improved);
    report.daily.trained = true;
else
    report.daily.trained = false;
    report.daily.improved = false;
    report.daily.orig = calcMetrics([], []);
    report.daily.new = [];
    report.daily.series = [];
    fprintf('跳过 daily 模型训练。\n');
end

% ===================== 月度模型：完全基于月均值预测3表 =====================
if trainMonthly
    try
        % 新月度模型使用 cfg.predictorCols（含新增列）
        monthlyCfg_new = cfg;
        monthlyCfg_new.sheet = cfg.monthlySheet;
        [P_month_all, T_month_all, D_month_all] = readTargetBlock(monthlyCfg_new, opts.inputDir, ...
            cfg.monthlyStartRow, cfg.monthlyEndRow, true);
        [P_month_all, T_month_all, D_month_all] = keepCompleteSamples(P_month_all, T_month_all, D_month_all);
        [P_month_all, T_month_all, D_month_all] = sortSamplesByDate(P_month_all, T_month_all, D_month_all);

        trainMaskMonth = D_month_all >= trainStartNum & D_month_all <= trainEndNum;
        testMaskMonth  = D_month_all >= testStartNum & D_month_all <= testEndNum;

        P_month_train = P_month_all(:, trainMaskMonth);
        T_month_train = T_month_all(trainMaskMonth);
        D_month_train = D_month_all(trainMaskMonth);
        P_month_test  = P_month_all(:, testMaskMonth);
        T_month_test  = T_month_all(testMaskMonth);
        D_month_test  = D_month_all(testMaskMonth);

        if numel(T_month_train) < 10
            report.messages{end+1} = '月度训练样本不足（少于10个），跳过月度模型重训练。';
            fprintf('  monthly 训练样本不足，跳过。\n');
            report.monthly.trained = true;
            report.monthly.available = false;
        elseif isempty(T_month_test)
            report.messages{end+1} = '测试区间内没有月度观测样本，月度模型保持不变。';
            fprintf('  monthly 测试样本为空，保留原月度模型。\n');
            report.monthly.trained = true;
            report.monthly.available = false;
        else
            % 追加月份特征（sin/cos）
            P_month_train_aug = appendMonthFeatures(P_month_train, D_month_train);
            P_month_test_aug  = appendMonthFeatures(P_month_test, D_month_test);

            fprintf('训练 monthly 模型（基于月均值预测3表）…\n');
            newMonthly = trainOneModel(P_month_train_aug, T_month_train, opts, monthlyCfg_new, 'monthly');
            newMonthlyPredTest = predictWithModel(newMonthly, P_month_test_aug);
            report.monthly.new = calcMetrics(T_month_test, newMonthlyPredTest);

            % 原月度模型：使用 origPredictorCols（不含新增列）读取数据并预测
            monthlyCfg_orig = cfg;
            monthlyCfg_orig.sheet = cfg.monthlySheet;
            monthlyCfg_orig.predictorCols = cfg.origPredictorCols;
            [P_month_orig_all, ~, D_month_orig_all] = readTargetBlock(monthlyCfg_orig, opts.inputDir, ...
                cfg.monthlyStartRow, cfg.monthlyEndRow, true);
            [P_month_orig_all, D_month_orig_all] = keepCompleteFeatureSamples(P_month_orig_all, D_month_orig_all);
            [P_month_orig_all, D_month_orig_all] = sortFeaturesByDate(P_month_orig_all, D_month_orig_all);
            P_month_orig_all_aug = appendMonthFeatures(P_month_orig_all, D_month_orig_all);
            % 按测试日期对齐
            [~, idx_orig] = ismember(D_month_test, D_month_orig_all);
            if any(idx_orig == 0)
                missingDates = D_month_test(idx_orig == 0);
                shown = arrayfun(@(x) datestr(x, 'yyyy-mm-dd'), missingDates(1:min(5,end)), 'UniformOutput', false);
                error('原月度模型测试集缺少日期：%s', strjoin(shown, ', '));
            end
            P_month_orig_test = P_month_orig_all_aug(:, idx_orig);
            origMonthlyPredTest = predictWithModel(origMonthly, P_month_orig_test);
            report.monthly.orig = calcMetrics(T_month_test, origMonthlyPredTest);

            report.monthly.available = true;
            report.monthly.testN = numel(T_month_test);
            report.monthly.series = struct('dates', D_month_test, 'actual', T_month_test, ...
                'newPred', newMonthlyPredTest, 'origPred', origMonthlyPredTest);
            report.monthly.improved = isBetter(report.monthly.new, report.monthly.orig);
            fprintf('  monthly 新 RMSE=%.4g R2=%.4g | 原 RMSE=%.4g R2=%.4g | 更优=%d\n', ...
                report.monthly.new.rmse, report.monthly.new.R2, ...
                report.monthly.orig.rmse, report.monthly.orig.R2, report.monthly.improved);
        end
        report.monthly.trained = true;
    catch mErr
        report.messages{end+1} = sprintf('月度模型重训练出错：%s', mErr.message);
        fprintf('  monthly 出错：%s\n', mErr.message);
        report.monthly.trained = true;
        report.monthly.available = false;
    end
else
    fprintf('跳过 monthly 模型训练。\n');
    report.monthly.trained = false;
    report.monthly.available = false;
    report.monthly.improved = false;
    report.monthly.orig = calcMetrics([], []);
    report.monthly.new = [];
    report.monthly.series = [];
end

% ---------------- 生成候选文件 ----------------
dailyImproved = report.daily.trained && report.daily.improved;
monthlyImproved = report.monthly.trained && report.monthly.improved;

if dailyImproved || monthlyImproved
    if dailyImproved
        dailyStruct = mergeLearnedFields(origDaily, newDaily);
    else
        dailyStruct = origDaily;
    end
    if monthlyImproved
        monthlyStruct = mergeLearnedFields(origMonthly, newMonthly);
    else
        monthlyStruct = origMonthly;
    end
    candidateFile = fullfile(rootDir, ...
        sprintf('predict_fixed_air_quality_%s_candidate.m', tag));
    writeCandidateFile(predictFile, candidateFile, tag, dailyStruct, monthlyStruct);
    report.candidateFile = candidateFile;
    report.updated = true;
    fprintf('已生成候选文件：%s\n', candidateFile);
else
    fprintf('未生成候选文件（无任一训练尺度满足 RMSE 更低且 R² 更高）。\n');
end

% ---------------- 写对比表 ----------------
report.outputDir = fullfile(rootDir, 'outputs', sprintf('retrain_%s', tag), ...
    datestr(now, 'yyyymmdd_HHMMSS'));
if ~exist(report.outputDir, 'dir')
    mkdir(report.outputDir);
end
report.summaryFile = fullfile(report.outputDir, ...
    sprintf('retrain_%s_comparison.xlsx', tag));
writeComparisonWorkbook(report);
fprintf('对比结果已保存：%s\n', report.summaryFile);
end

% =====================================================================
% 辅助函数：写对比表
% =====================================================================
function writeComparisonWorkbook(report)
header = {'scale', 'metric', 'new_model', 'original_model', 'better'};
rows = header;

if isfield(report.daily, 'trained') && report.daily.trained
    rows = appendScaleRows(rows, 'daily', report.daily);
end

if isfield(report.monthly, 'trained') && report.monthly.trained && report.monthly.available
    rows = appendScaleRows(rows, 'monthly', report.monthly);
end

meta = {
    'indicator', report.indicator;
    'train_period', [report.trainStart ' ~ ' report.trainEnd];
    'test_period',  [report.testStart ' ~ ' report.testEnd];
    'daily_train_n', report.dailyTrainN;
    'daily_test_n',  report.dailyTestN;
    'new_predictor_cols', strjoin(report.newPredictorCols, ',');
    'original_daily_predictor_cols', strjoin(report.originalDailyPredictorCols, ',');
    'original_monthly_predictor_cols', strjoin(report.originalMonthlyPredictorCols, ',');
    'daily_updated', tf2str(report.daily.trained && report.daily.improved);
    'monthly_updated', tf2str(report.monthly.trained && report.monthly.improved);
    'candidate_file', report.candidateFile;
    'criterion', 'new RMSE < original RMSE AND new R2 > original R2'};

writeCellSheet(report.summaryFile, meta, 'summary');
writeCellSheet(report.summaryFile, rows, 'metrics_compare');
if ~isempty(report.messages)
    notes = [{'note'}; report.messages(:)];
    writeCellSheet(report.summaryFile, notes, 'notes');
end

if isfield(report.daily, 'series') && ~isempty(report.daily.series)
    writeSeriesSheet(report.summaryFile, 'daily_series', report.daily.series);
end
if report.monthly.trained && report.monthly.available && isfield(report.monthly, 'series') && ~isempty(report.monthly.series)
    writeSeriesSheet(report.summaryFile, 'monthly_series', report.monthly.series);
end
end

function rows = appendScaleRows(rows, scaleName, s)
metricNames = {'rmse', 'mae', 'mapePercent', 'R', 'R2'};
metricLabels = {'RMSE', 'MAE', 'MAPE_percent', 'R', 'R2'};
higherBetter = [false false false true true];
for i = 1:numel(metricNames)
    nv = s.new.(metricNames{i});
    ov = s.orig.(metricNames{i});
    if higherBetter(i)
        better = pickName(nv > ov, nv < ov);
    else
        better = pickName(nv < ov, nv > ov);
    end
    rows(end+1, :) = {scaleName, metricLabels{i}, nv, ov, better};
end
end

function name = pickName(newWins, origWins)
if newWins
    name = 'new';
elseif origWins
    name = 'original';
else
    name = 'tie';
end
end

function s = tf2str(tf)
if tf
    s = 'yes';
else
    s = 'no';
end
end

function writeSeriesSheet(file, sheet, s)
d = s.dates(:); a = s.actual(:); np = s.newPred(:); op = s.origPred(:);
n = numel(d);
data = cell(n + 1, 4);
data(1, :) = {'date', 'actual', 'new_prediction', 'original_prediction'};
for i = 1:n
    if isfinite(d(i))
        ds = datestr(d(i), 'yyyy-mm-dd');
    else
        ds = '';
    end
    data(i + 1, :) = {ds, a(i), np(i), op(i)};
end
writeCellSheet(file, data, sheet);
end

% =====================================================================
% 训练与预测
% =====================================================================
function model = trainOneModel(P_train, T_train, opts, cfg, modelType)
[p_train, ps_input] = mapminmax(P_train, 0, 1);
[t_train, ps_output] = mapminmax(T_train, 0, 1);

fobj = @(x) timeSeriesBlockedFitness(x, P_train', T_train', opts.type, ...
    opts.kernel, opts.timeSeriesCvFolds, opts.timeSeriesCvInitialTrainRatio);

setRandomSeed(opts.randomSeed);
[bestFitness, bestX] = SCA(opts.SearchAgents_no, opts.Max_iteration, ...
    opts.lb, opts.ub, opts.dim, fobj);

[alpha, b] = trainlssvm({p_train', t_train', opts.type, ...
    bestX(1), bestX(2), opts.kernel});

model.modelType = modelType;
model.createdAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
model.gamma = bestX(1);
model.sig2 = bestX(2);
model.bestFitness = bestFitness;
model.type = opts.type;
model.kernel = opts.kernel;
model.alpha = alpha(:);
model.b = b;
model.pTrain = p_train';
model.tTrain = t_train(:);
model.psInput = ps_input;
model.psOutput = ps_output;
model.predictorCols = cfg.predictorCols;
model.targetCol = cfg.targetCol;
model.dateCol = cfg.dateCol;
model.monthlyWindowDays = opts.monthlyWindowDays;
model.monthlyWindowStepDays = opts.monthlyWindowStepDays;
end

function prediction = predictWithModel(model, P)
if isempty(P)
    prediction = [];
    return;
end
p = mapminmax('apply', P, model.psInput);
t = simlssvm({model.pTrain, model.tTrain, model.type, ...
    model.gamma, model.sig2, model.kernel}, {model.alpha, model.b}, p')';
prediction = mapminmax('reverse', t, model.psOutput);
end

function tf = isBetter(newMetrics, origMetrics)
tf = isfinite(newMetrics.rmse) && isfinite(origMetrics.rmse) && ...
    isfinite(newMetrics.R2) && isfinite(origMetrics.R2) && ...
    newMetrics.rmse < origMetrics.rmse && newMetrics.R2 > origMetrics.R2;
end

% =====================================================================
% 读取原嵌入模型结构
% =====================================================================
function model = embeddedModelStruct(predictFile, modelType)
txt = fileread(predictFile);
startTok = 'function model = fixedEmbeddedModel(modelType)';
sigIdx = strfind(txt, startTok);
if isempty(sigIdx)
    error('retrain_air_quality_model:NoEmbeddedBlock', ...
        '在 %s 中找不到 fixedEmbeddedModel。', predictFile);
end
sigIdx = sigIdx(1);
after = txt(sigIdx + length(startTok):end);
nl = regexp(after, '\nfunction ', 'once');
if isempty(nl)
    block = after;
else
    block = after(1:nl - 1);
end
body = regexprep(block, '\s*end\s*$', '');
modelType = lower(char(modelType));
model = [];
eval(body);
if isempty(model)
    error('retrain_air_quality_model:EmbeddedParseFailed', ...
        '解析嵌入模型失败 (%s)。', modelType);
end
end

% =====================================================================
% 生成候选文件
% =====================================================================
function writeCandidateFile(predictFile, candidateFile, tag, dailyStruct, monthlyStruct)
txt = fileread(predictFile);
startTok = 'function model = fixedEmbeddedModel(modelType)';
sigIdx = strfind(txt, startTok);
sigIdx = sigIdx(1);
after = txt(sigIdx:end);
nl = regexp(after, '\nfunction ', 'once');
if isempty(nl)
    error('retrain_air_quality_model:BlockBoundary', '无法定位嵌入模型代码块边界。');
end
blockEndAbs = sigIdx - 1 + nl;
pre = txt(1:sigIdx - 1);
post = txt(blockEndAbs:end);

newFunc = buildEmbeddedFunction(tag, dailyStruct, monthlyStruct);

banner = sprintf(['%% NOTE: 本文件由 retrain_air_quality_model 自动生成（候选模型）。\n' ...
    '%% 生成时间：%s\n' ...
    '%% 经验证后，可将其重命名/覆盖为 predict_fixed_air_quality_%s.m 以正式启用。\n'], ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS'), tag);

newTxt = [banner newFunc];
fid = fopen(candidateFile, 'w');
if fid < 0
    error('retrain_air_quality_model:WriteFailed', '无法写入候选文件：%s', candidateFile);
end
fwrite(fid, newTxt);
fclose(fid);
end

function s = buildEmbeddedFunction(tag, dailyStruct, monthlyStruct)
lines = {};
lines{end+1} = 'function model = fixedEmbeddedModel(modelType)';
lines{end+1} = 'modelType = lower(char(modelType));';
lines{end+1} = 'if strcmp(modelType, ''daily'')';
lines{end+1} = serializeModelStruct(dailyStruct);
lines{end+1} = 'elseif strcmp(modelType, ''monthly'')';
lines{end+1} = serializeModelStruct(monthlyStruct);
lines{end+1} = 'else';
lines{end+1} = sprintf(['    error(''predict_fixed_air_quality_%s:UnknownModelType'', ' ...
    '''modelType must be daily or monthly.'');'], tag);
lines{end+1} = 'end';
lines{end+1} = 'end';
s = strjoin(lines, sprintf('\n'));
end

function out = mergeLearnedFields(origStruct, newStruct)
out = origStruct;
learned = {'gamma', 'sig2', 'alpha', 'b', 'pTrain', 'tTrain', ...
    'psInput', 'psOutput', 'createdAt'};
for i = 1:numel(learned)
    f = learned{i};
    if isfield(newStruct, f)
        out.(f) = newStruct.(f);
    end
end
end

function s = serializeModelStruct(m)
fns = fieldnames(m);
L = cell(1, numel(fns));
for i = 1:numel(fns)
    L{i} = serializeField(['model.' fns{i}], m.(fns{i}));
end
s = strjoin(L, sprintf('\n'));
end

function line = serializeField(lhs, v)
if isstruct(v)
    sf = fieldnames(v);
    sub = cell(1, numel(sf));
    for j = 1:numel(sf)
        sub{j} = serializeField([lhs '.' sf{j}], v.(sf{j}));
    end
    line = strjoin(sub, sprintf('\n'));
elseif ischar(v)
    line = sprintf('    %s = ''%s'';', lhs, strrep(v, '''', ''''''));
elseif islogical(v)
    line = sprintf('    %s = %s;', lhs, mat2str(v));
elseif iscell(v)
    line = sprintf('    %s = %s;', lhs, cellColsToCode(v));
elseif isnumeric(v)
    if isscalar(v)
        line = sprintf('    %s = %s;', lhs, num17(v));
    elseif isempty(v)
        line = sprintf('    %s = [];', lhs);
    else
        line = sprintf('    %s = %s;', lhs, mat2str(v, 17));
    end
else
    error('retrain_air_quality_model:UnsupportedField', ...
        '字段 %s 的类型无法序列化。', lhs);
end
end

function s = num17(v)
s = sprintf('%.17g', double(v));
end

function s = cellColsToCode(cols)
parts = cell(1, numel(cols));
for i = 1:numel(cols)
    parts{i} = sprintf('''%s''', char(cols{i}));
end
s = ['{' strjoin(parts, ', ') '}'];
end

% =====================================================================
% 配置与选项
% =====================================================================
function opts = defaultOptions(rootDir)
opts.SearchAgents_no = 30;
opts.Max_iteration = 30;
opts.dim = 2;
opts.lb = [0.001, 0.001];
opts.ub = [500, 100];
opts.type = 'function estimation';
opts.kernel = 'RBF_kernel';
opts.inputDir = rootDir;
opts.randomSeed = 20260528;
opts.timeSeriesCvFolds = 3;
opts.timeSeriesCvInitialTrainRatio = 0.6;
opts.monthlyWindowDays = 30;
opts.monthlyWindowStepDays = 1;
end

function opts = mergeOptions(opts, userOpts)
fn = fieldnames(userOpts);
for i = 1:numel(fn)
    if ~isempty(userOpts.(fn{i}))
        opts.(fn{i}) = userOpts.(fn{i});
    end
end
end

function [cfg, tag] = indicatorConfig(indicator)
key = upper(strtrim(char(indicator)));
switch key
    case 'PM2.5'
        % 新模型增加 O 列
        cfg.predictorCols = {'D', 'E', 'G', 'I', 'J', 'K', 'M', 'O'};
        cfg.origPredictorCols = {'D', 'E', 'G', 'I', 'J', 'K', 'M'};  % 原模型不含 O
        cfg.targetCol = 'H'; tag = 'pm25';
    case 'PM10'
        cfg.predictorCols = {'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'M'};
        cfg.origPredictorCols = {'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'M'};
        cfg.targetCol = 'G'; tag = 'pm10';
    case 'SO2'
        cfg.predictorCols = {'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M'};
        cfg.origPredictorCols = {'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M'};
        cfg.targetCol = 'C'; tag = 'so2';
    case 'NO2'
        cfg.predictorCols = {'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M'};
        cfg.origPredictorCols = {'C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M'};
        cfg.targetCol = 'D'; tag = 'no2';
    case 'CO'
        cfg.predictorCols = {'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'M'};
        cfg.origPredictorCols = {'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'M'};
        cfg.targetCol = 'E'; tag = 'co';
    case 'O3'
        % 新模型增加 N 列
        cfg.predictorCols = {'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'M', 'N'};
        cfg.origPredictorCols = {'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'M'};  % 原模型不含 N
        cfg.targetCol = 'F'; tag = 'o3';
    otherwise
        error('retrain_air_quality_model:UnknownIndicator', ...
            '未知污染物：%s', indicator);
end
cfg.workbook = 'dataall_NEW.xlsx';
cfg.sheet = '日均值预测1';
cfg.dailySheet = '日均值预测1';
cfg.monthlySheet = '月均值预测3';
cfg.dateCol = 'B';
cfg.dailyStartRow = 2;
cfg.dailyEndRow = 1673;
cfg.monthlyStartRow = 2;
cfg.monthlyEndRow = 85;
end

function rootDir = scriptRootDir()
scriptPath = which('retrain_air_quality_model_V3');
if isempty(scriptPath)
    scriptPath = mfilename('fullpath');
end
rootDir = fileparts(scriptPath);
end

function d = parseDateValue(value)
if isnumeric(value) && isscalar(value)
    d = value;
elseif isa(value, 'datetime')
    d = datenum(value);
elseif ischar(value) || (isstring(value) && isscalar(value))
    d = datenum(char(value));
else
    error('retrain_air_quality_model:BadDate', '无法解析日期。');
end
end

% =====================================================================
% 新旧模型输入列对齐工具
% =====================================================================
function cols = originalPredictorCols(model, cfg, modelType)
if isfield(model, 'predictorCols') && ~isempty(model.predictorCols)
    cols = model.predictorCols;
elseif isfield(cfg, 'origPredictorCols') && ~isempty(cfg.origPredictorCols)
    cols = cfg.origPredictorCols;
else
    cols = cfg.predictorCols;
end
cols = reshape(cols, 1, []);
expectedCount = modelInputBaseFeatureCount(model, modelType);
if ~isempty(expectedCount) && numel(cols) ~= expectedCount
    if isfield(cfg, 'origPredictorCols') && numel(cfg.origPredictorCols) == expectedCount
        cols = cfg.origPredictorCols;
    else
        error('retrain_air_quality_model:OriginalPredictorMismatch', ...
            ['原 %s 模型需要 %d 个基础输入列，但当前解析到 %d 个 predictorCols。' ...
             '请检查 predict_fixed_air_quality 文件中的 predictorCols。'], ...
            char(modelType), expectedCount, numel(cols));
    end
end
end

function count = modelInputBaseFeatureCount(model, modelType)
count = [];
if isfield(model, 'pTrain') && ~isempty(model.pTrain)
    count = size(model.pTrain, 2);
elseif isfield(model, 'psInput') && isfield(model.psInput, 'xmin')
    count = numel(model.psInput.xmin);
end
if isempty(count)
    return;
end
if strcmpi(char(modelType), 'monthly')
    count = count - 2;
end
if count < 0
    count = [];
end
end

function [P, D] = keepCompleteFeatureSamples(P, D)
if isempty(P)
    return;
end
if nargin < 2 || isempty(D)
    D = NaN(1, size(P, 2));
end
valid = all(isfinite(P), 1) & isfinite(D);
P = P(:, valid);
D = D(:, valid);
end

function [P, D] = sortFeaturesByDate(P, D)
if isempty(P) || isempty(D) || any(~isfinite(D))
    return;
end
[D, order] = sort(D);
P = P(:, order);
end

function P_target = alignFeaturesByDate(P_source, D_source, D_target, label)
if isempty(D_target)
    P_target = zeros(size(P_source, 1), 0);
    return;
end
[tf, loc] = ismember(D_target, D_source);
if ~all(tf)
    missingDates = D_target(~tf);
    shownCount = min(numel(missingDates), 5);
    shown = cell(1, shownCount);
    for i = 1:shownCount
        shown{i} = datestr(missingDates(i), 'yyyy-mm-dd');
    end
    error('retrain_air_quality_model:FeatureDateMismatch', ...
        '%s缺少 %d 个日期的原模型输入特征，例如：%s', ...
        label, sum(~tf), strjoin(shown, ', '));
end
P_target = P_source(:, loc);
end

% =====================================================================
% 数据读取与时序工具
% =====================================================================
function [P, T, D] = readTargetBlock(cfg, inputDir, rowStart, rowEnd, readTarget)
workbookPath = resolveInputWorkbook(inputDir, cfg.workbook);
if isempty(workbookPath)
    error('retrain_air_quality_model:InputWorkbookMissing', ...
        '找不到输入工作簿：%s', fullfile(inputDir, cfg.workbook));
end
sampleCount = rowEnd - rowStart + 1;
P = readNumericColumns(workbookPath, cfg.sheet, cfg.predictorCols, rowStart, rowEnd)';
D = readDateRange(workbookPath, cfg.sheet, cfg.dateCol, rowStart, rowEnd);
if readTarget
    targetRange = [cfg.targetCol num2str(rowStart) ':' cfg.targetCol num2str(rowEnd)];
    T = readNumericRange(workbookPath, cfg.sheet, targetRange, sampleCount, 1)';
else
    T = [];
end
end

function data = readNumericColumns(workbookPath, sheetName, colNames, rowStart, rowEnd)
sampleCount = rowEnd - rowStart + 1;
data = NaN(sampleCount, numel(colNames));
for i = 1:numel(colNames)
    colName = char(colNames{i});
    rangeText = [colName num2str(rowStart) ':' colName num2str(rowEnd)];
    colData = readNumericRange(workbookPath, sheetName, rangeText, sampleCount, 1);
    data(:, i) = colData(:, 1);
end
end

function dates = readDateRange(workbookPath, sheetName, dateCol, rowStart, rowEnd)
sampleCount = rowEnd - rowStart + 1;
rangeText = [dateCol num2str(rowStart) ':' dateCol num2str(rowEnd)];
dates = NaN(1, sampleCount);
if exist('readcell', 'file') || exist('readcell', 'builtin')
    raw = readcell(workbookPath, 'Sheet', sheetName, 'Range', rangeText);
    raw = raw(:);
    for i = 1:min(numel(raw), sampleCount)
        dates(i) = normalizeExcelDate(raw{i});
    end
else
    raw = readNumericRange(workbookPath, sheetName, rangeText, sampleCount, 1);
    for i = 1:sampleCount
        dates(i) = normalizeExcelDate(raw(i));
    end
end
end

function d = normalizeExcelDate(value)
d = NaN;
if isnumeric(value)
    if isscalar(value) && isfinite(value)
        if value > 10000 && value < 100000
            d = value + datenum(1899, 12, 30);
        elseif value > 700000
            d = value;
        end
    end
elseif isa(value, 'datetime')
    d = datenum(value);
elseif ischar(value) || isstring(value)
    textValue = char(value);
    if ~isempty(textValue)
        try
            d = datenum(textValue);
        catch
            d = NaN;
        end
    end
end
end

function data = readNumericRange(workbookPath, sheetName, rangeText, expectedRows, expectedCols)
if exist('readmatrix', 'file') || exist('readmatrix', 'builtin')
    data = readmatrix(workbookPath, 'Sheet', sheetName, 'Range', rangeText);
else
    data = xlsread(workbookPath, sheetName, rangeText);
end
data = padNumericBlock(data, expectedRows, expectedCols);
end

function data = padNumericBlock(data, expectedRows, expectedCols)
if isempty(data)
    data = NaN(expectedRows, expectedCols);
    return;
end
[rowCount, colCount] = size(data);
if rowCount < expectedRows
    data(rowCount + 1:expectedRows, :) = NaN;
end
if colCount < expectedCols
    data(:, colCount + 1:expectedCols) = NaN;
end
data = data(1:expectedRows, 1:expectedCols);
end

function [P, T, D] = keepCompleteSamples(P, T, D)
if nargin < 3
    D = NaN(1, numel(T));
end
valid = all(isfinite(P), 1) & isfinite(T);
P = P(:, valid);
T = T(:, valid);
D = D(:, valid);
end

function [P, D, T] = keepCompleteFuture(P, D, T)
if isempty(P)
    if nargin < 3
        T = [];
    end
    return;
end
if nargin < 2
    D = NaN(1, size(P, 2));
end
if nargin < 3
    T = [];
end
valid = all(isfinite(P), 1);
P = P(:, valid);
D = D(:, valid);
if ~isempty(T)
    T = T(:, valid);
end
end

function [P, T, D] = sortSamplesByDate(P, T, D)
if nargin < 3 || isempty(D) || any(~isfinite(D))
    return;
end
[D, order] = sort(D);
P = P(:, order);
T = T(:, order);
end

function [P_month, T_month, D_month] = makeSlidingMonthlySamples(P, T, D, windowDays, stepDays)
sampleCount = numel(T);
if sampleCount < windowDays
    P_month = []; T_month = []; D_month = [];
    return;
end
sampleIndexes = windowDays:stepDays:sampleCount;
P_month = NaN(size(P, 1), numel(sampleIndexes));
T_month = NaN(1, numel(sampleIndexes));
D_month = NaN(1, numel(sampleIndexes));
for i = 1:numel(sampleIndexes)
    endIndex = sampleIndexes(i);
    startIndex = endIndex - windowDays + 1;
    P_month(:, i) = mean(P(:, startIndex:endIndex), 2);
    T_month(i) = mean(T(startIndex:endIndex));
    D_month(i) = D(endIndex);
end
end

function P = appendMonthFeatures(P, D)
if isempty(P)
    return;
end
months = datevec(D(:));
monthNumber = months(:, 2)';
P = [P; sin(2 * pi * monthNumber / 12); cos(2 * pi * monthNumber / 12)];
end

function fitness = timeSeriesBlockedFitness(position, xTrainRaw, yTrainRaw, type, ...
    kernel, foldCount, initialTrainRatio)
gamma = position(1);
sig2 = position(2);
sampleCount = size(xTrainRaw, 1);
if sampleCount < 10
    fitness = Inf; return;
end
initialTrainCount = floor(sampleCount * initialTrainRatio);
initialTrainCount = max(5, min(initialTrainCount, sampleCount - 1));
validationCount = sampleCount - initialTrainCount;
foldCount = max(1, min(foldCount, validationCount));
foldEdges = unique(round(linspace(initialTrainCount + 1, sampleCount + 1, foldCount + 1)));
if numel(foldEdges) < 2
    fitness = Inf; return;
end
sumSquaredError = 0;
predictionCount = 0;
try
    for fold = 1:(numel(foldEdges) - 1)
        validationStart = foldEdges(fold);
        validationEnd = foldEdges(fold + 1) - 1;
        if validationStart > validationEnd
            continue;
        end
        trainEnd = validationStart - 1;
        xFoldTrainRaw = xTrainRaw(1:trainEnd, :);
        yFoldTrainRaw = yTrainRaw(1:trainEnd, :);
        xFoldValidationRaw = xTrainRaw(validationStart:validationEnd, :);
        yFoldValidationRaw = yTrainRaw(validationStart:validationEnd, :);
        [pFoldTrain, psInputFold] = mapminmax(xFoldTrainRaw', 0, 1);
        pFoldValidation = mapminmax('apply', xFoldValidationRaw', psInputFold);
        [tFoldTrain, psOutputFold] = mapminmax(yFoldTrainRaw', 0, 1);
        xFoldTrain = pFoldTrain';
        yFoldTrain = tFoldTrain';
        xFoldValidation = pFoldValidation';
        [alpha, b] = trainlssvm({xFoldTrain, yFoldTrain, type, gamma, sig2, kernel});
        yPredNormalized = simlssvm({xFoldTrain, yFoldTrain, type, ...
            gamma, sig2, kernel}, {alpha, b}, xFoldValidation);
        yPred = mapminmax('reverse', yPredNormalized', psOutputFold)';
        errors = yFoldValidationRaw - yPred;
        sumSquaredError = sumSquaredError + sum(errors .^ 2);
        predictionCount = predictionCount + numel(errors);
    end
    if predictionCount == 0
        fitness = Inf;
    else
        fitness = sqrt(sumSquaredError / predictionCount);
    end
catch
    fitness = Inf;
end
end

function metrics = calcMetrics(actual, predicted)
actual = actual(:)';
predicted = predicted(:)';
err = predicted - actual;
metrics.mae = mean(abs(err));
metrics.mse = mean(err .^ 2);
metrics.rmse = sqrt(metrics.mse);
nonZero = actual ~= 0;
if any(nonZero)
    metrics.mapePercent = mean(abs(err(nonZero) ./ actual(nonZero))) * 100;
else
    metrics.mapePercent = NaN;
end
if numel(actual) > 1
    r = corrcoef(actual, predicted);
    metrics.R = r(1, 2);
else
    metrics.R = NaN;
end
den = norm(actual - mean(actual)) ^ 2;
if den == 0
    metrics.R2 = NaN;
else
    metrics.R2 = 1 - norm(actual - predicted) ^ 2 / den;
end
end

function setRandomSeed(seedValue)
if exist('rng', 'file') || exist('rng', 'builtin')
    rng(seedValue, 'twister');
else
    rand('seed', seedValue);
end
end

function workbookPath = resolveInputWorkbook(inputDir, defaultWorkbook)
workbookPath = fullfile(inputDir, defaultWorkbook);
if exist(workbookPath, 'file')
    return;
end
[~, baseName, ext] = fileparts(defaultWorkbook);
if strcmpi(baseName, 'dataall_NEW')
    if strcmpi(ext, '.xlsx')
        alternatives = {'dataall_NEW.xls'};
    elseif strcmpi(ext, '.xls')
        alternatives = {'dataall_NEW.xlsx'};
    else
        alternatives = {'dataall_NEW.xlsx', 'dataall_NEW.xls'};
    end
    for i = 1:numel(alternatives)
        candidate = fullfile(inputDir, alternatives{i});
        if exist(candidate, 'file')
            workbookPath = candidate;
            return;
        end
    end
end
workbookPath = '';
end

function writeCellSheet(filename, data, sheetName)
if exist('writecell', 'file') || exist('writecell', 'builtin')
    writecell(data, filename, 'Sheet', sheetName);
else
    xlswrite(filename, data, sheetName);
end
end