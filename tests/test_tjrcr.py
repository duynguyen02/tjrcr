import pytest
import pandas as pd
from tjwb import TJWBResult
from tjrcr import is_comprehensive_regulation
from tjrcr.tjrcr import _prepare_dataframe_for_P_n_calculation, _is_12_months_each_year, _is_greater_than_10_years, \
    _calculate_P_n, _year, _month, _delta_t


@pytest.fixture
def sample_dataframe():
    data = {
        'datetime': pd.date_range(start='2010-01-01', periods=120, freq='ME'),
        'inflow_speed': range(120),
        'outflow_speed': range(120, 240)
    }
    return pd.DataFrame(data)


@pytest.fixture
def tjwb_result(sample_dataframe):
    return TJWBResult(
        datetime=sample_dataframe['datetime'],
        inflow_speed=sample_dataframe['inflow_speed'],
        outflow_speed=sample_dataframe['outflow_speed'],
        components_outflow_speed={}
    )


def test_is_12_months_each_year(sample_dataframe):
    result = _is_12_months_each_year(sample_dataframe)
    assert result


def test_is_greater_than_10_years(sample_dataframe):
    result = _is_greater_than_10_years(sample_dataframe)
    assert result


def test_prepare_dataframe_for_P_n_calculation(sample_dataframe):
    df = _prepare_dataframe_for_P_n_calculation(sample_dataframe)
    assert _year in df.columns
    assert _month in df.columns
    assert _delta_t in df.columns
    assert df[_delta_t].iloc[0] == 0


def test_calculate_P_n(sample_dataframe):
    df = _prepare_dataframe_for_P_n_calculation(sample_dataframe)
    V_c = 1000
    P_n = _calculate_P_n(df, V_c)
    assert isinstance(P_n, float)
    assert 0 <= P_n <= 100


def test_is_comprehensive_regulation(tjwb_result):
    eps = 1.0
    P = 80.0
    V_c = 1000.0
    result = is_comprehensive_regulation(tjwb_result, eps, P, V_c)
    assert isinstance(result, bool)


def test_is_comprehensive_regulation_invalid_years(tjwb_result):
    # Test with less than 10 years of data
    data = {
        'datetime': pd.date_range(start='2015-01-01', periods=24, freq='ME'),
        'inflow_speed': range(24),
        'outflow_speed': range(24, 48)
    }
    df = pd.DataFrame(data)
    tjwb_result = TJWBResult(
        datetime=df['datetime'],
        inflow_speed=df['inflow_speed'],
        outflow_speed=df['outflow_speed'],
        components_outflow_speed={}
    )

    with pytest.raises(ValueError, match="Requires at least 10 years."):
        is_comprehensive_regulation(tjwb_result, eps=1.0, P=80.0, V_c=1000.0)


def test_is_comprehensive_regulation_invalid_months(tjwb_result):
    # Test with less than 12 months of data per year
    data = {
        'datetime': pd.date_range(start='2010-01-01', periods=11, freq='ME'),
        'inflow_speed': range(11),
        'outflow_speed': range(11, 22)
    }
    df = pd.DataFrame(data)
    tjwb_result = TJWBResult(
        datetime=df['datetime'],
        inflow_speed=df['inflow_speed'],
        outflow_speed=df['outflow_speed'],
        components_outflow_speed={}
    )
    with pytest.raises(ValueError, match="Requires 12 months in each year."):
        is_comprehensive_regulation(tjwb_result, eps=1.0, P=80.0, V_c=1000.0, forced_gt_10_year=False)
