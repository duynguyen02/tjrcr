import pandas as pd
from tjwb import TJWBResult

# Base on TJWBResult
_datetime = 'datetime'
_inflow_speed = 'inflow_speed'
_outflow_speed = 'outflow_speed'
# Addition
_year = 'year'
_month = 'month'
_year_month = 'year_month'
_delta_t = 'delta_t'
_capacity_ht = 'capacity_ht'


def _is_12_months_each_year(df: pd.DataFrame):
    t_df = df.copy()
    t_df[_year] = t_df[_datetime].dt.year
    t_df[_month] = t_df[_datetime].dt.month
    months_per_year = t_df.groupby(_year)[_month].nunique()
    return (months_per_year == 12).all()


def _is_greater_than_10_years(df: pd.DataFrame):
    return len(df[_datetime].dt.year.unique()) >= 10


def _prepare_dataframe_for_P_n_calculation(_df: pd.DataFrame):
    df = _df.copy()
    df[_year] = df[_datetime].dt.year
    df[_month] = df[_datetime].dt.month

    df.set_index(_datetime, inplace=True)
    df = df.resample('ME').mean()
    df.reset_index(drop=True, inplace=True)

    df[_year] = df[_year].astype(int)
    df[_month] = df[_month].astype(int)

    df[_year_month] = pd.to_datetime(df[_year].astype(str) + '-' + df[_month].astype(str), format='%Y-%m')
    df[_delta_t] = df[_year_month].diff().dt.total_seconds().fillna(0)
    df = df.drop(columns=[_year_month])

    return df


def _calculate_P_n(
        _df: pd.DataFrame,
        V_c: float
):
    df = _df.copy()

    unique_years = df[_year].unique()
    enough_water_years = 0

    for y in unique_years:
        year_df = df.loc[df[_year] == y].copy()

        capacity_ht = []
        previous_capacity_ht = V_c
        for index, row in year_df.iterrows():
            previous_capacity_ht = abs(
                previous_capacity_ht + ((row[_outflow_speed] - row[_inflow_speed]) * row[_delta_t]) / 10 ** 6)
            capacity_ht.append(previous_capacity_ht)

        year_df[_capacity_ht] = capacity_ht
        if year_df[_capacity_ht].min() > V_c:
            enough_water_years += 1

    P_n = (enough_water_years / len(unique_years)) * 100

    return P_n


def is_comprehensive_regulation(
        tjwb_result: TJWBResult,
        eps: float,
        P: float,
        V_c: float,
        forced_gt_10_year: bool = True,
        forced_12_months_each_year: bool = True
):
    df = tjwb_result.to_dataframe()
    df[_datetime] = pd.to_datetime(df[_datetime])
    df = df.sort_values(by=_datetime)

    if forced_gt_10_year and not _is_greater_than_10_years(df):
        raise ValueError("Requires at least 10 years.")

    if forced_12_months_each_year and not _is_12_months_each_year(df):
        raise ValueError("Requires 12 months in each year.")

    df = _prepare_dataframe_for_P_n_calculation(df)

    P_n = _calculate_P_n(df, V_c)

    return (P_n - P) <= eps
