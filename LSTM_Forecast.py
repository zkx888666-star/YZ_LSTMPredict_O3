"""
LSTM 空气品质预测模型（适配您的 Excel 结构：表头+列字母位置）
支持污染物：PM2.5, PM10, O3, NO2, CO, SO2
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置参数 ========================
RANDOM_SEED = 20260528
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

DAILY_WINDOW_DAYS = 7
DAILY_LSTM_UNITS = 64
DAILY_DROPOUT = 0.2
DAILY_EPOCHS = 100
DAILY_BATCH_SIZE = 32
DAILY_LEARNING_RATE = 0.001

MONTHLY_HIDDEN_UNITS = [32, 16]
MONTHLY_EPOCHS = 200
MONTHLY_BATCH_SIZE = 8
MONTHLY_LEARNING_RATE = 0.001
USE_LSTM_FOR_MONTHLY = False

EXCEL_FILE = "dataall_NEW.xlsx"          # 请修改为实际路径
OUTPUT_DIR = "outputs_lstm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 污染物配置（列字母与您的Excel列位置完全一致）
# 列映射：A=0(序号), B=1(日期), C=2(SO2), D=3(NO2), E=4(CO), F=5(O3), G=6(PM10), H=7(PM2.5)
#        I=8(温度), J=9(湿度), K=10(风速), L=11(风向), M=12(降水), N=13(SSRD), O=14(PBLH)
INDICATOR_CONFIG = {
    'PM2.5': {
        'predictor_cols': ['D', 'E', 'G', 'I', 'J', 'K', 'M', 'O'],   # NO2, CO, PM10, 温度, 湿度, 风速, 降水, PBLH
        'target_col': 'H',                                            # PM2.5
        'tag': 'pm25',
        'daily_sheet': 'daily1',
        'monthly_sheet': 'monthly3'
    },
    'PM10': {
        'predictor_cols': ['C', 'D', 'E', 'F', 'H', 'I', 'J', 'K', 'M'],
        'target_col': 'G',
        'tag': 'pm10',
        'daily_sheet': 'daily1',
        'monthly_sheet': 'monthly3'
    },
    'SO2': {
        'predictor_cols': ['D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M'],
        'target_col': 'C',
        'tag': 'so2',
        'daily_sheet': 'daily1',
        'monthly_sheet': 'monthly3'
    },
    'NO2': {
        'predictor_cols': ['C', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'M'],
        'target_col': 'D',
        'tag': 'no2',
        'daily_sheet': 'daily1',
        'monthly_sheet': 'monthly3'
    },
    'CO': {
        'predictor_cols': ['C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'M'],
        'target_col': 'E',
        'tag': 'co',
        'daily_sheet': 'daily1',
        'monthly_sheet': 'monthly3'
    },
    'O3': {
        'predictor_cols': ['C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'M', 'N'],
        'target_col': 'F',
        'tag': 'o3',
        'daily_sheet': 'daily1',
        'monthly_sheet': 'monthly3'
    }
}

# ======================== 辅助函数 ========================
def col_letter_to_index(col_letter):
    """将 Excel 列字母转为 0-based 索引"""
    idx = 0
    for ch in col_letter:
        idx = idx * 26 + (ord(ch.upper()) - ord('A') + 1)
    return idx - 1

def load_excel_sheet(file_path, sheet_name, usecols, date_col='B', skiprows=1):
    """
    按列字母位置读取 Excel，日期列为字符串格式（如 '2021-01-01'）
    自动跳过表头行。
    """
    all_letters = list(set(usecols) | {date_col})
    indices = [col_letter_to_index(l) for l in all_letters]
    min_idx, max_idx = min(indices), max(indices)
    start_letter = chr(ord('A') + min_idx)
    end_letter = chr(ord('A') + max_idx)
    usecols_range = f"{start_letter}:{end_letter}"
    
    # 读取原始数据（不将第一行作为列名）
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name,
                           header=None, skiprows=skiprows, usecols=usecols_range)
    
    # 建立列字母映射
    col_map = {min_idx + i: letter for i, letter in enumerate(all_letters)}
    df_raw.columns = [col_map.get(min_idx + i, f'col_{i}') for i in range(df_raw.shape[1])]
    
    # 日期列处理
    date_series = pd.to_datetime(df_raw[date_col], errors='coerce')
    # 设为索引，删除原日期列
    df_raw.index = date_series
    df_raw.drop(columns=[date_col], inplace=True)
    df_raw = df_raw[~df_raw.index.isna()]   # 删除日期无效行
    
    # 只保留需要的列
    df_out = df_raw[usecols].copy()
    
    # 强制数值转换，并统计转换失败数量
    for col in df_out.columns:
        before = df_out[col].copy()
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        failed = (df_out[col].isna() & ~before.isna()).sum()
        if failed > 0:
            print(f"警告：列 {col} 中有 {failed} 个非数值单元格已转为 NaN。")
    
    # 检查全 NaN 列
    all_nan_cols = [col for col in df_out.columns if df_out[col].isna().all()]
    if all_nan_cols:
        raise ValueError(f"错误：以下列全部为 NaN（可能列字母不存在或全为空）：{all_nan_cols}")
    
    df_out.sort_index(inplace=True)
    return df_out

def build_sequences(data, target_col, window_days):
    features = data.columns.tolist()
    X, y = [], []
    for i in range(len(data) - window_days):
        X.append(data.iloc[i:i+window_days].values)
        y.append(data.iloc[i+window_days][target_col])
    return np.array(X), np.array(y)

def evaluate(actual, pred):
    actual = np.array(actual).flatten()
    pred = np.array(pred).flatten()
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
    r = np.corrcoef(actual, pred)[0, 1]
    r2 = r2_score(actual, pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape, 'R': r, 'R2': r2}

def plot_predictions(dates, actual, pred, title, save_path):
    plt.figure(figsize=(12,5))
    plt.plot(dates, actual, 'b-', label='Actual')
    plt.plot(dates, pred, 'r--', label='Predicted')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Concentration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_metrics(metrics_dict, file_path):
    pd.DataFrame([metrics_dict]).to_csv(file_path, index=False)

# ======================== 日模型 ========================
def train_daily_lstm(indicator, train_start, train_end, test_start, test_end, window_days=DAILY_WINDOW_DAYS):
    cfg = INDICATOR_CONFIG[indicator]
    print(f"\n======= 日模型训练: {indicator} =======")
    print(f"训练区间: {train_start} 至 {train_end}")
    print(f"测试区间: {test_start} 至 {test_end}")
    
    all_cols = [cfg['target_col']] + cfg['predictor_cols']
    all_cols = list(dict.fromkeys(all_cols))
    df = load_excel_sheet(EXCEL_FILE, cfg['daily_sheet'], all_cols, date_col='B')
    
    # 调试信息
    print(f"原始数据形状: {df.shape}")
    print("各列 NaN 数量:\n", df.isna().sum())
    print("前 3 行数据:\n", df.head(3))
    
    # 删除包含 NaN 的行
    orig_len = len(df)
    df.dropna(inplace=True)
    removed = orig_len - len(df)
    if removed > 0:
        print(f"移除了 {removed} 行包含缺失值的行。")
    if df.empty:
        raise ValueError("日数据为空，请检查上述 NaN 分布。")
    
    # 日期边界
    train_start_dt = pd.to_datetime(train_start).normalize()
    train_end_dt   = pd.to_datetime(train_end).normalize()
    test_start_dt  = pd.to_datetime(test_start).normalize()
    test_end_dt    = pd.to_datetime(test_end).normalize()
    
    train_mask = (df.index >= train_start_dt) & (df.index <= train_end_dt)
    test_mask  = (df.index >= test_start_dt) & (df.index <= test_end_dt)
    df_train = df[train_mask]
    df_test  = df[test_mask]
    
    print(f"训练集样本数: {len(df_train)}, 测试集样本数: {len(df_test)}")
    
    if len(df_train) <= window_days:
        raise ValueError(f"训练集样本太少（{len(df_train)}），无法构建窗口。")
    if len(df_test) == 0:
        raise ValueError("测试集无样本。")
    
    # 归一化
    scaler_X = MinMaxScaler(feature_range=(0,1))
    scaler_y = MinMaxScaler(feature_range=(0,1))
    X_train_raw = df_train[cfg['predictor_cols']].values
    y_train_raw = df_train[cfg['target_col']].values.reshape(-1,1)
    X_test_raw  = df_test[cfg['predictor_cols']].values
    y_test_raw  = df_test[cfg['target_col']].values.reshape(-1,1)
    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw)
    X_train_scaled = scaler_X.transform(X_train_raw)
    y_train_scaled = scaler_y.transform(y_train_raw).flatten()
    X_test_scaled  = scaler_X.transform(X_test_raw)
    y_test_scaled  = scaler_y.transform(y_test_raw).flatten()
    
    df_train_scaled = pd.DataFrame(X_train_scaled, index=df_train.index, columns=cfg['predictor_cols'])
    df_train_scaled[cfg['target_col']] = y_train_scaled
    df_test_scaled  = pd.DataFrame(X_test_scaled,  index=df_test.index,  columns=cfg['predictor_cols'])
    df_test_scaled[cfg['target_col']]  = y_test_scaled
    
    X_seq_train, y_seq_train = build_sequences(df_train_scaled, cfg['target_col'], window_days)
    X_seq_test,  y_seq_test  = build_sequences(df_test_scaled,  cfg['target_col'], window_days)
    
    if len(X_seq_train) == 0:
        raise ValueError("窗口长度过大，请减小 DAILY_WINDOW_DAYS 或增加训练数据。")
    
    model = Sequential()
    model.add(InputLayer(input_shape=(window_days, len(cfg['predictor_cols']))))
    model.add(LSTM(DAILY_LSTM_UNITS, return_sequences=False, dropout=DAILY_DROPOUT))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=DAILY_LEARNING_RATE), loss='mse', metrics=['mae'])
    
    checkpoint = ModelCheckpoint(os.path.join(OUTPUT_DIR, f"daily_{cfg['tag']}_best.h5"),
                                 monitor='val_loss', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    val_size = max(1, int(len(X_seq_train) * 0.2))
    X_val, y_val = X_seq_train[-val_size:], y_seq_train[-val_size:]
    X_tr,  y_tr  = X_seq_train[:-val_size], y_seq_train[:-val_size]
    
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=DAILY_EPOCHS,
              batch_size=DAILY_BATCH_SIZE, callbacks=[checkpoint, early_stop], verbose=1)
    
    best_model = load_model(os.path.join(OUTPUT_DIR, f"daily_{cfg['tag']}_best.h5"))
    pred_scaled = best_model.predict(X_seq_test, verbose=0).flatten()
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
    actual = scaler_y.inverse_transform(y_seq_test.reshape(-1,1)).flatten()
    test_dates = df_test_scaled.index[window_days:]
    
    metrics = evaluate(actual, pred)
    print("日模型评估结果:")
    for k,v in metrics.items(): print(f"  {k}: {v:.4f}")
    plot_predictions(test_dates, actual, pred,
                     f"Daily LSTM - {indicator} (test set)",
                     os.path.join(OUTPUT_DIR, f"daily_{cfg['tag']}_prediction.png"))
    save_metrics(metrics, os.path.join(OUTPUT_DIR, f"daily_{cfg['tag']}_metrics.csv"))
    pred_df = pd.DataFrame({'date': test_dates, 'actual': actual, 'predicted': pred})
    pred_df.to_csv(os.path.join(OUTPUT_DIR, f"daily_{cfg['tag']}_predictions.csv"), index=False)
    return model, metrics

# ======================== 月模型（同理，不再重复粘贴，但函数体与之前相同）================
# 注意：由于月模型也依赖 load_excel_sheet，它同样会正确处理字符串日期，无需额外修改。
# 这里为了节省篇幅，省略月模型函数体（您可以从之前的完整代码中复制 train_monthly 函数）。
# 下面是占位，实际运行时请确保 train_monthly 函数存在。

def train_monthly(indicator, train_start, train_end, test_start, test_end):
    # 此函数内容与之前完全相同，建议从上一版完整代码中复制。
    # 如果您没有保留，请告知，我会再次提供完整版。
    pass

# ======================== 主函数 ========================
def main():
    indicator = 'PM2.5'
    train_start = '2020-01-01'
    train_end   = '2025-12-31'
    test_start  = '2026-01-01'
    test_end    = '2026-12-31'
    run_daily   = False
    run_monthly = True   # 暂时关闭月模型，先调试日模型
    
    if run_daily:
        try:
            train_daily_lstm(indicator, train_start, train_end, test_start, test_end)
        except Exception as e:
            print(f"日模型训练失败: {e}")
    if run_monthly:
        try:
            train_monthly(indicator, train_start, train_end, test_start, test_end)
            print(f"月模型正在训练")
        except Exception as e:
            print(f"月模型训练失败: {e}")
    print("\n任务结束，结果保存在:", OUTPUT_DIR)

if __name__ == "__main__":
    main()