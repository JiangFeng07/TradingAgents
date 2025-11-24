import os
import requests
import pandas as pd
import json
from datetime import datetime
from io import StringIO

import numpy as np
from dateutil.relativedelta import relativedelta
import talib
import baostock as bs
from typing import Optional, Dict, Any

API_BASE_URL = "https://www.alphavantage.co/query"

def get_api_key() -> str:
    """Retrieve the API key for Alpha Vantage from environment variables."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY environment variable is not set.")
    return api_key

def format_datetime_for_api(date_input) -> str:
    """Convert various date formats to YYYYMMDDTHHMM format required by Alpha Vantage API."""
    if isinstance(date_input, str):
        # If already in correct format, return as-is
        if len(date_input) == 13 and 'T' in date_input:
            return date_input
        # Try to parse common date formats
        try:
            dt = datetime.strptime(date_input, "%Y-%m-%d")
            return dt.strftime("%Y%m%dT0000")
        except ValueError:
            try:
                dt = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
                return dt.strftime("%Y%m%dT%H%M")
            except ValueError:
                raise ValueError(f"Unsupported date format: {date_input}")
    elif isinstance(date_input, datetime):
        return date_input.strftime("%Y%m%dT%H%M")
    else:
        raise ValueError(f"Date must be string or datetime object, got {type(date_input)}")

class AlphaVantageRateLimitError(Exception):
    """Exception raised when Alpha Vantage API rate limit is exceeded."""
    pass

def _make_api_request(function_name: str, params: dict) -> dict | str:
    """Helper function to make API requests and handle responses.
    
    Raises:
        AlphaVantageRateLimitError: When API rate limit is exceeded
    """
    # Create a copy of params to avoid modifying the original
    api_params = params.copy()
    api_params.update({
        "function": function_name,
        "apikey": get_api_key(),
        "source": "trading_agents",
    })
    
    # Handle entitlement parameter if present in params or global variable
    current_entitlement = globals().get('_current_entitlement')
    entitlement = api_params.get("entitlement") or current_entitlement
    
    if entitlement:
        api_params["entitlement"] = entitlement
    elif "entitlement" in api_params:
        # Remove entitlement if it's None or empty
        api_params.pop("entitlement", None)
    
    response = requests.get(API_BASE_URL, params=api_params)
    response.raise_for_status()

    response_text = response.text
    
    # Check if response is JSON (error responses are typically JSON)
    try:
        response_json = json.loads(response_text)
        # Check for rate limit error
        if "Information" in response_json:
            info_message = response_json["Information"]
            if "rate limit" in info_message.lower() or "api key" in info_message.lower():
                raise AlphaVantageRateLimitError(f"Alpha Vantage rate limit exceeded: {info_message}")
    except json.JSONDecodeError:
        # Response is not JSON (likely CSV data), which is normal
        pass

    return response_text



def _filter_csv_by_date_range(csv_data: str, start_date: str, end_date: str) -> str:
    """
    Filter CSV data to include only rows within the specified date range.

    Args:
        csv_data: CSV string from Alpha Vantage API
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Filtered CSV string
    """
    if not csv_data or csv_data.strip() == "":
        return csv_data

    try:
        # Parse CSV data
        df = pd.read_csv(StringIO(csv_data))

        # Assume the first column is the date column (timestamp)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])

        # Filter by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        filtered_df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]

        # Convert back to CSV string
        return filtered_df.to_csv(index=False)

    except Exception as e:
        # If filtering fails, return original data with a warning
        print(f"Warning: Failed to filter CSV data by date range: {e}")
        return csv_data


class BaoStockDataFetcher:
    """Baostock数据获取器，封装登录和请求逻辑"""
    
    def __init__(self):
        self._logged_in = False
    
    def __enter__(self):
        self.login()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()
    
    def login(self) -> bool:
        """登录Baostock"""
        if not self._logged_in:
            result = bs.login()
            self._logged_in = result.error_code == '0'
            if not self._logged_in:
                raise ConnectionError(f"Baostock登录失败: {result.error_msg}")
        return self._logged_in
    
    def logout(self) -> None:
        """登出Baostock"""
        if self._logged_in:
            bs.logout()
            self._logged_in = False


def get_stock_history_k_data(
    stock_code: str, 
    start_date: str = '2024-01-01',
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    获取股票历史K线数据
    
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期，默认为当前日期
    
    Returns:
        pandas DataFrame包含股票数据
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 验证日期格式
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"日期格式错误: {e}")
    
    # 使用上下文管理器确保资源正确释放
    with BaoStockDataFetcher():
        rs = bs.query_history_k_data_plus(
            stock_code,
            "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date,
            end_date=end_date,
            frequency='d',
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            raise ConnectionError(f"数据查询失败: {rs.error_msg}")
        
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return pd.DataFrame(columns=rs.fields)
        
        return pd.DataFrame(data_list, columns=rs.fields)


class TechnicalIndicatorCalculator:
    """技术指标计算器"""
    
    # 指标参数配置
    INDICATOR_CONFIGS = {
        "close_50_sma": {"func": talib.SMA, "params": {"timeperiod": 50}},
        "close_200_sma": {"func": talib.SMA, "params": {"timeperiod": 200}},
        "close_10_ema": {"func": talib.EMA, "params": {"timeperiod": 10}},
        "macd": {"func": talib.MACD, "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}, "output_idx": 0},
        "macds": {"func": talib.MACD, "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}, "output_idx": 1},
        "macdh": {"func": talib.MACD, "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}, "output_idx": 2},
        "rsi": {"func": talib.RSI, "params": {"timeperiod": 14}},
        "boll":{'func':talib.BBANDS, "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0}, "output_idx":1},
        "boll_ub":{'func':talib.BBANDS, "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0}, "output_idx":2},
        "boll_lb":{'func':talib.BBANDS, "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2, "matype": 0}, "output_idx":0},
        "atr": {"func": talib.ATR, "params": {"timeperiod": 14}},
        "vwma": {"func": talib.WMA, "params": {"timeperiod": 30}},
    }
    
    
    @classmethod
    def calculate_indicator(cls, data: pd.DataFrame, indicator: str) -> pd.Series:
        """
        计算技术指标
        
        Args:
            data: 包含价格数据的DataFrame
            indicator: 指标名称
        
        Returns:
            计算后的指标序列
        """
        if indicator not in cls.INDICATOR_CONFIGS:
            raise ValueError(f"不支持的指标: {indicator}。支持的指标: {list(cls.INDICATOR_CONFIGS.keys())}")
        
        config = cls.INDICATOR_CONFIGS[indicator]
        
        if indicator not in ['atr']:
        
            # 确保close列是数值类型
            close_prices = data['close'].astype(float)
            if close_prices.isnull().any():
                raise ValueError("close列包含非数值数据")
            
            # 计算指标
            result = config["func"](close_prices, **config["params"])
            
        else:
            high_prices = pd.to_numeric(data['high'], errors='coerce')
            low_prices = pd.to_numeric(data['low'], errors='coerce')
            close_prices = pd.to_numeric(data['close'], errors='coerce')
            if high_prices.isnull().any() or low_prices.isnull().any() or close_prices.isnull().any():
                raise ValueError("high, low, close列包含非数值数据")
            result = config["func"](high_prices, low_prices, close_prices, **config["params"])
        
        # 处理多输出指标（如MACD）
        if "output_idx" in config:
            if isinstance(result, tuple):
                return result[config["output_idx"]]
            else:
                raise ValueError(f"指标{indicator}预期返回元组，但得到{type(result)}")
        
        return result


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理DataFrame：转换数据类型、排序等
    
    Args:
        df: 原始DataFrame
    
    Returns:
        处理后的DataFrame
    """
    df = df.copy()
    
    # 按日期排序
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    # 转换数值列
    numeric_columns = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def validate_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """
    验证日期范围是否有效
    
    Args:
        df: 包含日期数据的DataFrame
        start_date: 开始日期
        end_date: 结束日期
    """
    if df.empty:
        raise ValueError("数据框为空")
    
    if 'date' not in df.columns:
        raise ValueError("数据框不包含日期列")
    
    date_series = pd.to_datetime(df['date'])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    if start_dt > end_dt:
        raise ValueError("开始日期不能晚于结束日期")
    
    available_dates = date_series[(date_series >= start_dt) & (date_series <= end_dt)]
    if available_dates.empty:
        raise ValueError(f"在{start_date}到{end_date}范围内没有可用数据")


def _make_baostock_request(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
    start_date: str = '2024-01-01'
) -> pd.DataFrame:
    """
    获取股票数据并计算技术指标
    
    Args:
        symbol: 股票代码
        indicator: 技术指标名称
        cur_date: 当前日期
        look_back_days: 回溯天数
        start_date: 数据开始日期
    
    Returns:
        包含指标数据的DataFrame
    """
    # 参数验证
    if look_back_days <= 0:
        raise ValueError("look_back_days必须大于0")
    
    # 计算日期范围
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before_date = curr_date_dt - relativedelta(days=look_back_days)
    before_date_str = before_date.strftime("%Y-%m-%d")
    
    try:
        # 获取原始数据
        tmp_df = get_stock_history_k_data(
            stock_code=symbol, 
            start_date=start_date, 
            end_date=curr_date
        )
        
        if tmp_df.empty:
            raise ValueError(f"未获取到股票{symbol}的数据")
        
        # 数据预处理
        tmp_df = preprocess_dataframe(tmp_df)
        
        # 验证日期范围
        validate_date_range(tmp_df, before_date_str, curr_date)
        
        # 计算技术指标
        indicator_values = TechnicalIndicatorCalculator.calculate_indicator(tmp_df, indicator)
        tmp_df[indicator] = indicator_values
        
        # 筛选指定日期范围的数据
        result_df = tmp_df[
            (tmp_df['date'] >= before_date_str) & 
            (tmp_df['date'] <= curr_date)
        ].reset_index(drop=True)
        result_df.rename(columns={'date': 'time'}, inplace=True)
        return result_df
        
    except Exception as e:
        raise RuntimeError(f"处理股票{symbol}指标{indicator}时出错: {e}")


def batch_calculate_indicators(
    symbol: str,
    indicators: list,
    cur_date: str,
    look_back_days: int,
    start_date: str = '2024-01-01'
) -> pd.DataFrame:
    """
    批量计算多个技术指标
    
    Args:
        symbol: 股票代码
        indicators: 指标列表
        cur_date: 当前日期
        look_back_days: 回溯天数
        start_date: 数据开始日期
    
    Returns:
        包含所有指标的DataFrame
    """
    # 获取基础数据（只获取一次）
    curr_date_dt = datetime.strptime(cur_date, "%Y-%m-%d")
    before_date = curr_date_dt - relativedelta(days=look_back_days)
    before_date_str = before_date.strftime("%Y-%m-%d")
    
    base_df = get_stock_history_k_data(
        stock_code=symbol, 
        start_date=start_date, 
        end_date=cur_date
    )
    
    if base_df.empty:
        raise ValueError(f"未获取到股票{symbol}的数据")
    
    base_df = preprocess_dataframe(base_df)
    validate_date_range(base_df, before_date_str, cur_date)
    
    # 计算所有指标
    for indicator in indicators:
        try:
            indicator_values = TechnicalIndicatorCalculator.calculate_indicator(base_df, indicator)
            base_df[indicator] = indicator_values
        except Exception as e:
            print(f"计算指标{indicator}时出错: {e}")
            base_df[indicator] = np.nan
    
    # 返回指定日期范围的数据
    result_df = base_df[
        (base_df['date'] >= before_date_str) & 
        (base_df['date'] <= cur_date)
    ].reset_index(drop=True)
    
    return result_df

