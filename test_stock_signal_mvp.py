import pytest
import pandas as pd
from unittest.mock import patch
from stock_signal_mvp import StockSignalPipeline

def test_fetch_data_success():
    """Test successful data fetching"""
    with patch('yfinance.download') as mock_download:
        df = pd.DataFrame({'Close': [100, 101, 102]}, index=pd.date_range('2023-01-01', periods=3))
        mock_download.return_value = df
        pipeline = StockSignalPipeline('AAPL', '2023-01-01', '2023-01-03')
        result = pipeline.fetch_data()
        assert not result.empty
        assert list(result['Close']) == [100, 101, 102]

def test_fetch_data_empty():
    """Test handling of empty data"""
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = pd.DataFrame()
        pipeline = StockSignalPipeline('INVALID', '2023-01-01', '2023-01-03')
        result = pipeline.fetch_data()
        assert result.empty

def test_calculate_moving_averages():
    """Test moving average calculation"""
    df = pd.DataFrame({'Close': [10, 20, 30, 40, 50]})
    pipeline = StockSignalPipeline('AAPL', '2023-01-01', '2023-01-05', short_window=2, long_window=3)
    pipeline.data = df.copy()
    result = pipeline.calculate_moving_averages()
    
    assert 'Short_MA' in result.columns
    assert 'Long_MA' in result.columns
    
    # Test short MA calculation (2-period)
    assert result['Short_MA'].iloc[1] == pytest.approx(15.0)  # (10+20)/2
    assert result['Short_MA'].iloc[2] == pytest.approx(25.0)  # (20+30)/2
    
    # Test long MA calculation (3-period)
    assert result['Long_MA'].iloc[2] == pytest.approx(20.0)  # (10+20+30)/3
    assert result['Long_MA'].iloc[3] == pytest.approx(30.0)  # (20+30+40)/3

def test_generate_signals_buy():
    """Test Buy signal generation when short MA > long MA"""
    df = pd.DataFrame({
        'Short_MA': [105, 110, 115],
        'Long_MA':  [100, 100, 100]
    })
    pipeline = StockSignalPipeline('AAPL', '2023-01-01', '2023-01-03')
    pipeline.data = df.copy()
    result = pipeline.generate_signals()
    
    # All signals should be 'Buy' since short MA > long MA
    assert all(result['Signal'] == 'Buy')

def test_generate_signals_sell():
    """Test Sell signal generation when short MA < long MA"""
    df = pd.DataFrame({
        'Short_MA': [95, 90, 85],
        'Long_MA':  [100, 100, 100]
    })
    pipeline = StockSignalPipeline('AAPL', '2023-01-01', '2023-01-03')
    pipeline.data = df.copy()
    result = pipeline.generate_signals()
    
    # All signals should be 'Sell' since short MA < long MA
    assert all(result['Signal'] == 'Sell')

def test_generate_signals_mixed():
    """Test mixed signals with varying relationships"""
    df = pd.DataFrame({
        'Short_MA': [105, 100, 95, 110, 100],
        'Long_MA':  [100, 100, 100, 100, 100]
    })
    pipeline = StockSignalPipeline('AAPL', '2023-01-01', '2023-01-05')
    pipeline.data = df.copy()
    result = pipeline.generate_signals()
    
    # Index 0: 105 > 100 -> Buy
    assert result['Signal'].iloc[0] == 'Buy'
    # Index 1: 100 = 100 -> Hold (or whatever logic handles equality)
    # Index 2: 95 < 100 -> Sell
    assert result['Signal'].iloc[2] == 'Sell'
    # Index 3: 110 > 100 -> Buy
    assert result['Signal'].iloc[3] == 'Buy'

def test_generate_signals_edge_cases():
    """Test edge cases with equal values"""
    df = pd.DataFrame({
        'Short_MA': [100, 100, 100],
        'Long_MA':  [100, 100, 100]
    })
    pipeline = StockSignalPipeline('AAPL', '2023-01-01', '2023-01-03')
    pipeline.data = df.copy()
    result = pipeline.generate_signals()
    
    # When short MA = long MA, should be Hold
    assert all(result['Signal'] == 'Hold')

def test_integration():
    """Test the full pipeline"""
    with patch('yfinance.download') as mock_download:
        # Create mock data
        mock_data = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2023-01-01', periods=5))
        mock_download.return_value = mock_data
        
        # Test full pipeline
        pipeline = StockSignalPipeline('AAPL', '2023-01-01', '2023-01-05', short_window=2, long_window=3)
        pipeline.fetch_data()
        pipeline.calculate_moving_averages()
        result = pipeline.generate_signals()
        
        assert 'Signal' in result.columns
        assert len(result) == 5 