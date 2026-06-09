function results = simple_air_quality_lstm_cnn_leadtime(userOpts)
% 多步预测分析：固定模型结构与超参数，改变预测提前量（lead steps），
% 分别训练模型并评估性能，输出“预测时效 vs. 精度”曲线。
% 支持分别为日数据和月数据设置不同的提前量列表。

if nargin < 1 || isempty(userOpts)
    userOpts = struct();
end

opts = defaultOptions();
opts = mergeOptions(opts, userOpts);
setRandomSeed(opts.randomSeed);

if ~exist(opts.outputDir, 'dir')
    mkdir(opts.outputDir);
end

indicators = {'PM2.5', 'O3'};
scales = {'daily', 'monthly'};
modelTypes = {'LSTM', 'CNN'};

% 获取日/月不同的提前量列表
dailyLeadSteps = opts.dailyLeadStepsList;
monthlyLeadSteps = opts.monthlyLeadStepsList;

summaryRows = {};
resultIndex = 0;
results = struct([]);

% 存储集成模型指标用于绘图
ensembleStorage = containers.Map();

fprintf('=== 多步预测分析（中心化预处理）===\n');
fprintf('日数据提前量列表：'); fprintf('%d ', dailyLeadSteps); fprintf('\n');
fprintf('月数据提前量列表：'); fprintf('%d ', monthlyLeadSteps); fprintf('\n');
fprintf('训练集：%s ~ %s；测试集：%s ~ %s\n', ...
    opts.trainStart, opts.trainEnd, opts.testStart, opts.testEnd);

for i = 1:numel(indicators)
    indicator = indicators{i};
    cfg = indicatorConfig(indicator);
    for s = 1:numel(scales)
        scaleName = scales{s};
        % 根据尺度选择提前量列表
        if strcmpi(scaleName, 'daily')
            leadSteps = dailyLeadSteps;
        else
            leadSteps = monthlyLeadSteps;
        end
        
        % 加载一次原始数据
        data = loadAirQualityData(opts.workbook, cfg, scaleName);
        data = data(isfinite(data.target) & all(isfinite(data.predictors), 2), :);
        data = sortrows(data, 'date');
        
        for l = 1:numel(leadSteps)
            leadStep = leadSteps(l);
            fprintf('\n========== %s %s, 提前量 = %d ==========\n', ...
                indicator, scaleName, leadStep);
            
            singleResults = struct();
            
            for m = 1:numel(modelTypes)
                modelType = modelTypes{m};
                fprintf('--- 训练 %s ---\n', modelType);
                one = trainAndEvaluateOne(data, cfg, opts, scaleName, modelType, leadStep);
                
                resultIndex = resultIndex + 1;
                results(resultIndex).indicator = indicator;
                results(resultIndex).scale = scaleName;
                results(resultIndex).leadStep = leadStep;
                results(resultIndex).modelType = modelType;
                results(resultIndex).metrics = one.metrics;
                results(resultIndex).predictionFile = one.predictionFile;
                results(resultIndex).figureFile = one.figureFile;
                results(resultIndex).lossCurveFile = one.lossCurveFile;
                
                summaryRows(end+1, :) = {indicator, scaleName, leadStep, modelType, ...
                    one.metrics.RMSE, one.metrics.MAE, one.metrics.MAPE, ...
                    one.metrics.R, one.metrics.R2, one.predictionFile, one.figureFile, one.lossCurveFile};
                
                singleResults.(modelType).testDates = one.testDates;
                singleResults.(modelType).actual = one.actual;
                singleResults.(modelType).predicted = one.predicted;
            end
            
            % 集成平均
            if isfield(singleResults, 'LSTM') && isfield(singleResults, 'CNN')
                ensemble = computeEnsembleAverage(singleResults, indicator, scaleName, leadStep, opts);
                summaryRows(end+1, :) = {indicator, scaleName, leadStep, 'EnsembleAvg', ...
                    ensemble.metrics.RMSE, ensemble.metrics.MAE, ensemble.metrics.MAPE, ...
                    ensemble.metrics.R, ensemble.metrics.R2, ensemble.predictionFile, ensemble.figureFile, ''};
                
                resultIndex = resultIndex + 1;
                results(resultIndex).indicator = indicator;
                results(resultIndex).scale = scaleName;
                results(resultIndex).leadStep = leadStep;
                results(resultIndex).modelType = 'EnsembleAvg';
                results(resultIndex).metrics = ensemble.metrics;
                results(resultIndex).predictionFile = ensemble.predictionFile;
                results(resultIndex).figureFile = ensemble.figureFile;
                results(resultIndex).lossCurveFile = '';
                
                % 存储集成指标用于绘图
                key = sprintf('%s_%s', indicator, scaleName);
                if ~isKey(ensembleStorage, key)
                    ensembleStorage(key) = struct('leadSteps', [], 'RMSE', [], 'MAE', [], 'R2', []);
                end
                tmp = ensembleStorage(key);
                tmp.leadSteps(end+1) = leadStep;
                tmp.RMSE(end+1) = ensemble.metrics.RMSE;
                tmp.MAE(end+1) = ensemble.metrics.MAE;
                tmp.R2(end+1) = ensemble.metrics.R2;
                ensembleStorage(key) = tmp;
            end
        end
    end
end

% 保存汇总表
summaryTable = cell2table(summaryRows, 'VariableNames', ...
    {'indicator', 'scale', 'leadStep', 'modelType', 'RMSE', 'MAE', 'MAPE_percent', ...
     'R', 'R2', 'predictionFile', 'figureFile', 'lossCurveFile'});
summaryFile = fullfile(opts.outputDir, 'summary_leadtime_metrics.xlsx');
writetable(summaryTable, summaryFile);
fprintf('\n汇总指标已保存：%s\n', summaryFile);

% 绘制预测时效 vs 性能曲线（集成模型）
plotLeadTimePerformance(ensembleStorage, opts.outputDir);

fprintf('全部完成。\n');
end

% =====================================================================
% 单个模型训练与测试（支持提前量 leadStep）
% =====================================================================
function out = trainAndEvaluateOne(data, cfg, opts, scaleName, modelType, leadStep)
trainStart = datetime(opts.trainStart, 'InputFormat', 'yyyy-MM-dd');
trainEnd = datetime(opts.trainEnd, 'InputFormat', 'yyyy-MM-dd');
testStart = datetime(opts.testStart, 'InputFormat', 'yyyy-MM-dd');
testEnd = datetime(opts.testEnd, 'InputFormat', 'yyyy-MM-dd');

trainMask = data.date >= trainStart & data.date <= trainEnd;
testMask = data.date >= testStart & data.date <= testEnd;
trainData = data(trainMask, :);
testData = data(testMask, :);

if strcmpi(scaleName, 'daily')
    windowSize = opts.dailyWindow;
else
    windowSize = opts.monthlyWindow;
end

% 检查数据量是否足够（考虑 leadStep 偏移）
minTrainLen = windowSize + leadStep;
if height(trainData) <= minTrainLen
    error('训练样本太少：%s %s 提前量 %d 需要至少 %d 行，实际 %d 行。', ...
        cfg.indicator, scaleName, leadStep, minTrainLen+1, height(trainData));
end
minTestLen = windowSize + leadStep;
if height(testData) <= minTestLen
    error('测试样本太少：%s %s 提前量 %d 需要至少 %d 行，实际 %d 行。', ...
        cfg.indicator, scaleName, leadStep, minTestLen+1, height(testData));
end

% 构造序列（包含 leadStep 偏移）
[XTrain, YTrain, scalerX, scalerY] = makeSequencesCentered(trainData, windowSize, leadStep, scaleName, [], []);
[XTest, YTestScaled, ~, ~, testDates] = makeSequencesCentered(testData, windowSize, leadStep, scaleName, scalerX, scalerY);

layers = makeLayers(modelType, numel(cfg.predictorCols), opts);
trainOpts = trainingOptions('adam', ...
    'MaxEpochs', opts.maxEpochs, ...
    'MiniBatchSize', opts.miniBatchSize, ...
    'InitialLearnRate', opts.learningRate, ...
    'GradientThreshold', 1, ...
    'Shuffle', 'never', ...
    'Verbose', false, ...
    'Plots', 'none');

[net, trainingInfo] = trainNetwork(XTrain, YTrain, layers, trainOpts);

% 损失曲线
baseNameLoss = sprintf('%s_%s_%s_lead%d_loss', lower(cfg.tag), scaleName, lower(modelType), leadStep);
lossCurveFile = fullfile(opts.outputDir, [baseNameLoss '.png']);
plotLossCurve(trainingInfo, lossCurveFile, cfg.indicator, scaleName, modelType, leadStep);

% 预测与反中心化
predScaled = predict(net, XTest, 'MiniBatchSize', opts.miniBatchSize);
actual = reverseCenter(YTestScaled(:), scalerY, testDates, scaleName);
predicted = reverseCenter(predScaled(:), scalerY, testDates, scaleName);
metrics = calcMetrics(actual, predicted);

fprintf('Lead %d: RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%, R=%.4f, R2=%.4f\n', ...
    leadStep, metrics.RMSE, metrics.MAE, metrics.MAPE, metrics.R, metrics.R2);

% 保存预测表
baseName = sprintf('%s_%s_%s_lead%d', lower(cfg.tag), scaleName, lower(modelType), leadStep);
predictionFile = fullfile(opts.outputDir, [baseName '_predictions.xlsx']);
predictionTable = table(testDates(:), actual(:), predicted(:), ...
    'VariableNames', {'date', 'actual', 'predicted'});
writetable(predictionTable, predictionFile);

% 绘制预测对比图
figureFile = fullfile(opts.outputDir, [baseName '_prediction.png']);
plotPrediction(testDates, actual, predicted, ...
    sprintf('%s %s %s (lead=%d)', cfg.indicator, scaleName, modelType, leadStep), figureFile);

% 保存模型
modelFile = fullfile(opts.outputDir, [baseName '_net.mat']);
save(modelFile, 'net', 'scalerX', 'scalerY', 'cfg', 'opts', 'metrics');

out.metrics = metrics;
out.predictionFile = predictionFile;
out.figureFile = figureFile;
out.modelFile = modelFile;
out.lossCurveFile = lossCurveFile;
out.testDates = testDates;
out.actual = actual;
out.predicted = predicted;
end

% =====================================================================
% 序列构造（中心化版本，支持提前量 leadStep）
% =====================================================================
function [X, Y, scalerX, scalerY, yDates] = makeSequencesCentered(data, windowSize, leadStep, scaleName, scalerX, scalerY)
% data: table with fields date, target, predictors
% leadStep: 预测未来几步（1 表示窗口之后第一步，2 表示第二步...）
XRaw = data.predictors;
YRaw = data.target;
dates = data.date;

% 计算或复用中心化参数
if isempty(scalerX)
    scalerX = computeMeanCenter(XRaw, dates, scaleName);
end
if isempty(scalerY)
    scalerY = computeMeanCenter(YRaw, dates, scaleName);
end

% 中心化
XCentered = centerData(XRaw, scalerX, dates, scaleName);
YCentered = centerData(YRaw, scalerY, dates, scaleName);

% 构造滑动窗口，目标偏移为 windowSize + leadStep - 1
sequenceCount = height(data) - windowSize - (leadStep - 1);
if sequenceCount <= 0
    error('数据长度不足以构造提前量 %d 的序列。数据行数 %d，窗口 %d，leadStep %d', ...
        leadStep, height(data), windowSize, leadStep);
end
X = cell(sequenceCount, 1);
Y = NaN(sequenceCount, 1);
yDates = NaT(sequenceCount, 1);

for i = 1:sequenceCount
    X{i} = XCentered(i:i+windowSize-1, :)';
    targetIndex = i + windowSize + (leadStep - 1);
    Y(i) = YCentered(targetIndex);
    yDates(i) = dates(targetIndex);
end
end

function centerStruct = computeMeanCenter(values, dates, scaleName)
% values: N x d
if strcmpi(scaleName, 'daily')
    meanVal = mean(values, 1, 'omitnan');
    centerStruct.type = 'daily';
    centerStruct.meanOverall = meanVal;
else % monthly
    months = month(dates);
    monthlyMean = NaN(1, size(values,2), 12);
    for m = 1:12
        idx = (months == m);
        if any(idx)
            monthlyMean(1,:,m) = mean(values(idx,:), 1, 'omitnan');
        else
            monthlyMean(1,:,m) = mean(values, 1, 'omitnan');
        end
    end
    centerStruct.type = 'monthly';
    centerStruct.monthlyMean = monthlyMean;
end
end

function centered = centerData(values, centerStruct, dates, scaleName)
if strcmpi(scaleName, 'daily')
    centered = values - centerStruct.meanOverall;
else
    months = month(dates);
    centered = zeros(size(values));
    for i = 1:size(values,1)
        m = months(i);
        centered(i,:) = values(i,:) - centerStruct.monthlyMean(1,:,m);
    end
end
end

function reversed = reverseCenter(centered, centerStruct, dates, scaleName)
if strcmpi(scaleName, 'daily')
    reversed = centered + centerStruct.meanOverall;
else
    months = month(dates);
    reversed = zeros(size(centered));
    for i = 1:size(centered,1)
        m = months(i);
        reversed(i,:) = centered(i,:) + centerStruct.monthlyMean(1,:,m);
    end
end
if size(reversed,2) == 1
    reversed = reversed(:);
end
end

% =====================================================================
% 简单平均集成（支持提前量）
% =====================================================================
function ensemble = computeEnsembleAverage(singleResults, indicator, scaleName, leadStep, opts)
datesLSTM = singleResults.LSTM.testDates;
datesCNN = singleResults.CNN.testDates;
if ~isequal(datesLSTM, datesCNN)
    warning('Lead %d: LSTM 和 CNN 测试日期不一致，取交集平均。', leadStep);
    [commonDates, idxL, idxC] = intersect(datesLSTM, datesCNN);
    actual = singleResults.LSTM.actual(idxL);
    predLSTM = singleResults.LSTM.predicted(idxL);
    predCNN = singleResults.CNN.predicted(idxC);
else
    commonDates = datesLSTM;
    actual = singleResults.LSTM.actual;
    predLSTM = singleResults.LSTM.predicted;
    predCNN = singleResults.CNN.predicted;
end

predEnsemble = (predLSTM + predCNN) / 2;
metricsEnsemble = calcMetrics(actual, predEnsemble);

fprintf('Ensemble Lead %d: RMSE=%.4f, MAE=%.4f, MAPE=%.2f%%, R=%.4f, R2=%.4f\n', ...
    leadStep, metricsEnsemble.RMSE, metricsEnsemble.MAE, metricsEnsemble.MAPE, ...
    metricsEnsemble.R, metricsEnsemble.R2);

baseName = sprintf('%s_%s_ensemble_avg_lead%d', lower(strrep(indicator, '.', '')), scaleName, leadStep);
predictionFile = fullfile(opts.outputDir, [baseName '_predictions.xlsx']);
predTable = table(commonDates(:), actual(:), predEnsemble(:), ...
    'VariableNames', {'date', 'actual', 'predicted'});
writetable(predTable, predictionFile);

figureFile = fullfile(opts.outputDir, [baseName '_prediction.png']);
plotPrediction(commonDates, actual, predEnsemble, ...
    sprintf('%s %s Ensemble Avg (Lead=%d)', indicator, scaleName, leadStep), figureFile);

ensemble.metrics = metricsEnsemble;
ensemble.predictionFile = predictionFile;
ensemble.figureFile = figureFile;
end

% =====================================================================
% 绘制“预测时效 vs. 性能指标”曲线（针对集成模型）
% =====================================================================
function plotLeadTimePerformance(ensembleStorage, outputDir)
keys = ensembleStorage.keys();
if isempty(keys)
    warning('无集成模型数据，无法绘制预测时效曲线。');
    return;
end

for k = 1:numel(keys)
    key = keys{k};
    data = ensembleStorage(key);
    % 按 leadStep 排序
    [leadSorted, idx] = sort(data.leadSteps);
    rmseSorted = data.RMSE(idx);
    maeSorted = data.MAE(idx);
    r2Sorted = data.R2(idx);
    
    fig = figure('Visible', 'off', 'Position', [100,100,800,600]);
    subplot(2,2,1);
    plot(leadSorted, rmseSorted, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
    xlabel('Lead Step'); ylabel('RMSE'); grid on;
    title(sprintf('%s - RMSE vs Lead Time', strrep(key, '_', ' ')));
    
    subplot(2,2,2);
    plot(leadSorted, maeSorted, 'r-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
    xlabel('Lead Step'); ylabel('MAE'); grid on;
    title(sprintf('%s - MAE vs Lead Time', strrep(key, '_', ' ')));
    
    subplot(2,2,3);
    plot(leadSorted, r2Sorted, 'g-^', 'LineWidth', 1.5, 'MarkerFaceColor', 'g');
    xlabel('Lead Step'); ylabel('R²'); grid on;
    ylim([-0.5, 1]);
    title(sprintf('%s - R² vs Lead Time', strrep(key, '_', ' ')));
    
    subplot(2,2,4);
    % 归一化后比较
    rmse_norm = (rmseSorted - min(rmseSorted)) / (max(rmseSorted)-min(rmseSorted)+eps);
    mae_norm = (maeSorted - min(maeSorted)) / (max(maeSorted)-min(maeSorted)+eps);
    r2_norm = (r2Sorted - min(r2Sorted)) / (max(r2Sorted)-min(r2Sorted)+eps);
    plot(leadSorted, rmse_norm, 'b-o', leadSorted, mae_norm, 'r-s', leadSorted, r2_norm, 'g-^', 'LineWidth', 1.5);
    xlabel('Lead Step'); ylabel('Normalized Metric'); grid on;
    legend('RMSE', 'MAE', 'R²', 'Location', 'best');
    title('Normalized Metrics Comparison');
    
    sgtitle(sprintf('Performance vs Lead Time (%s)', strrep(key, '_', ' ')), 'FontSize', 12);
    
    saveFile = fullfile(outputDir, sprintf('leadtime_performance_%s.png', key));
    saveas(fig, saveFile);
    close(fig);
    fprintf('已保存预测时效曲线：%s\n', saveFile);
end
end

% =====================================================================
% 模型定义、数据读取等辅助函数
% =====================================================================
function layers = makeLayers(modelType, featureCount, opts)
switch upper(modelType)
    case 'LSTM'
        layers = [
            sequenceInputLayer(featureCount, 'Name', 'input')
            lstmLayer(opts.lstmUnits, 'OutputMode', 'last', 'Name', 'lstm')
            dropoutLayer(opts.dropout, 'Name', 'dropout')
            fullyConnectedLayer(1, 'Name', 'fc')
            regressionLayer('Name', 'regression')];
    case 'CNN'
        layers = [
            sequenceInputLayer(featureCount, 'Name', 'input')
            convolution1dLayer(3, opts.cnnFilters, 'Padding', 'same', 'Name', 'conv1')
            reluLayer('Name', 'relu1')
            convolution1dLayer(3, opts.cnnFilters, 'Padding', 'same', 'Name', 'conv2')
            reluLayer('Name', 'relu2')
            globalAveragePooling1dLayer('Name', 'gap')
            fullyConnectedLayer(1, 'Name', 'fc')
            regressionLayer('Name', 'regression')];
    otherwise
        error('未知模型类型：%s', modelType);
end
end

function plotLossCurve(trainingInfo, savePath, indicator, scaleName, modelType, leadStep)
if isempty(trainingInfo) || ~isfield(trainingInfo, 'TrainingLoss')
    warning('无训练损失信息，无法绘制损失曲线。');
    return;
end
loss = trainingInfo.TrainingLoss;
iterations = 1:length(loss);
fig = figure('Visible', 'off');
plot(iterations, loss, 'b-', 'LineWidth', 1.2);
grid on;
xlabel('Iteration');
ylabel('Training Loss');
title(sprintf('%s %s %s (lead=%d) Loss Curve', indicator, scaleName, modelType, leadStep));
saveas(fig, savePath);
close(fig);
end

function data = loadAirQualityData(workbook, cfg, scaleName)
if strcmpi(scaleName, 'daily')
    sheetName = cfg.dailySheet;
else
    sheetName = cfg.monthlySheet;
end
raw = readcell(workbook, 'Sheet', sheetName);
if size(raw, 1) < 2
    error('工作表 %s 数据为空或只有表头。', sheetName);
end
rowStart = 2;
rowEnd = size(raw, 1);
rowCount = rowEnd - rowStart + 1;
dates = NaT(rowCount, 1);
target = NaN(rowCount, 1);
predictors = NaN(rowCount, numel(cfg.predictorCols));
for r = rowStart:rowEnd
    outRow = r - rowStart + 1;
    dates(outRow) = parseExcelDate(raw{r, excelColToNum(cfg.dateCol)});
    target(outRow) = parseNumber(raw{r, excelColToNum(cfg.targetCol)});
    for c = 1:numel(cfg.predictorCols)
        predictors(outRow, c) = parseNumber(raw{r, excelColToNum(cfg.predictorCols{c})});
    end
end
valid = ~isnat(dates);
data = table(dates(valid), target(valid), predictors(valid, :), ...
    'VariableNames', {'date', 'target', 'predictors'});
fprintf('已读取 %s / %s：%d 行，有效日期 %d 行。\n', workbook, sheetName, rowCount, height(data));
end

function cfg = indicatorConfig(indicator)
switch upper(indicator)
    case 'PM2.5'
        cfg.indicator = 'PM2.5';
        cfg.predictorCols = {'D', 'E', 'G', 'I', 'J', 'K', 'M', 'O'};
        cfg.targetCol = 'H';
        cfg.tag = 'pm25';
    case 'O3'
        cfg.indicator = 'O3';
        cfg.predictorCols = {'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'M', 'N'};
        cfg.targetCol = 'F';
        cfg.tag = 'o3';
    otherwise
        error('本程序只处理 PM2.5 与 O3。');
end
cfg.dateCol = 'B';
cfg.dailySheet = '日均值预测1';
cfg.monthlySheet = '月均值预测3';
end

function opts = defaultOptions()
opts.workbook = 'dataall_NEW.xlsx';
opts.outputDir = fullfile(pwd, 'outputs_leadtime_analysis');
opts.trainStart = '2020-01-01';
opts.trainEnd = '2024-12-31';
opts.testStart = '2025-01-01';
opts.testEnd = '2025-07-31';
opts.dailyWindow = 7;
opts.monthlyWindow = 3;
opts.lstmUnits = 52;
opts.cnnFilters = 40;
opts.dropout = 0.15;
opts.maxEpochs = 80;
opts.miniBatchSize = 32;
opts.learningRate = 0.0001;
opts.randomSeed = 20260528;
% 分别为日数据和月数据设置不同的提前量列表
opts.dailyLeadStepsList = [1, 3, 5, 7, 14];    % 日：1,3,5,7,14天
opts.monthlyLeadStepsList = [1, 2, 3];      % 月：1,2,3,4个月
end

function opts = mergeOptions(opts, userOpts)
fields = fieldnames(userOpts);
for i = 1:numel(fields)
    opts.(fields{i}) = userOpts.(fields{i});
end
end

function n = excelColToNum(col)
col = upper(char(col));
n = 0;
for i = 1:numel(col)
    n = n * 26 + double(col(i)) - double('A') + 1;
end
end

function value = parseNumber(x)
if isnumeric(x) && isscalar(x)
    value = double(x);
elseif islogical(x)
    value = double(x);
elseif ischar(x) || isstring(x)
    value = str2double(string(x));
else
    value = NaN;
end
end

function d = parseExcelDate(x)
if isa(x, 'datetime')
    d = x;
elseif isnumeric(x) && isscalar(x) && isfinite(x)
    d = datetime(x, 'ConvertFrom', 'excel');
elseif ischar(x) || isstring(x)
    textValue = strtrim(string(x));
    if strlength(textValue) == 0 || ismissing(textValue)
        d = NaT;
    else
        try
            d = datetime(textValue, 'InputFormat', 'yyyy-MM-dd');
        catch
            d = datetime(textValue);
        end
    end
else
    d = NaT;
end
end

function metrics = calcMetrics(actual, predicted)
actual = actual(:);
predicted = predicted(:);
err = predicted - actual;
metrics.MAE = mean(abs(err), 'omitnan');
metrics.RMSE = sqrt(mean(err .^ 2, 'omitnan'));
nonZero = actual ~= 0 & isfinite(actual) & isfinite(predicted);
if any(nonZero)
    metrics.MAPE = mean(abs(err(nonZero) ./ actual(nonZero)), 'omitnan') * 100;
else
    metrics.MAPE = NaN;
end
valid = isfinite(actual) & isfinite(predicted);
if nnz(valid) > 1
    r = corrcoef(actual(valid), predicted(valid));
    metrics.R = r(1, 2);
else
    metrics.R = NaN;
end
den = sum((actual(valid) - mean(actual(valid))).^2);
if den == 0 || isempty(den)
    metrics.R2 = NaN;
else
    metrics.R2 = 1 - sum((actual(valid) - predicted(valid)).^2) / den;
end
end

function plotPrediction(dates, actual, predicted, titleText, savePath)
fig = figure('Visible', 'off');
plot(dates, actual, 'b-', 'LineWidth', 1.2);
hold on;
plot(dates, predicted, 'r--', 'LineWidth', 1.2);
grid on;
legend({'Actual', 'Predicted'}, 'Location', 'best');
xlabel('Date');
ylabel('Concentration');
title(titleText, 'Interpreter', 'none');
saveas(fig, savePath);
close(fig);
end

function setRandomSeed(seedValue)
if exist('rng', 'file') || exist('rng', 'builtin')
    rng(seedValue, 'twister');
else
    rand('seed', seedValue);
end
end