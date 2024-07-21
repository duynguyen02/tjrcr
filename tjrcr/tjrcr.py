import pandas as pd
from tjwb import WB


class TJRCR:
    def __init__(
            self,
            tjwb: WB,
            ets_df: pd.DataFrame,
            V_c: float,
            V_h: float
    ):
        self.tjwb = tjwb
        self.ets_df = ets_df
        self.V_c = V_c
        self.V_h = V_h

        if V_h not in self.ets_df['storage'].values:
            raise ValueError(f"Capacity does not exist!: {V_h}")

        if V_c not in self.ets_df['storage'].values:
            raise ValueError(f"Capacity does not exist!: {V_c}")

    @staticmethod
    def _is_greater_than_10_years(df: pd.DataFrame):
        return len(df['timestamp'].dt.year.unique()) >= 10

    @staticmethod
    def _is_12_months_each_year(df: pd.DataFrame):
        t_df = df.copy()
        t_df['year'] = t_df['timestamp'].dt.year
        t_df['month'] = t_df['timestamp'].dt.month
        months_per_year = t_df.groupby('year')['month'].nunique()
        return (months_per_year == 12).all()

    @staticmethod
    def _prepare_df_for_cal_P_n(
            pre_df: pd.DataFrame,
    ):
        df = pre_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month

        df = df[['timestamp', 'Q_out_total', 'Q_in', 'year', 'month']]

        df.set_index('timestamp', inplace=True)
        df = df.resample('ME').mean()
        df.reset_index(drop=True, inplace=True)

        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)

        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str), format='%Y-%m')
        df['delta_t'] = df['date'].diff().dt.total_seconds().fillna(0)
        df = df.drop(columns=['date'])

        return df

    @staticmethod
    def _calculate_P_n(
            pre_df: pd.DataFrame,
            V_c: float
    ):
        df = pre_df.copy()

        unique_years = df['year'].unique()
        enough_water_years = 0

        for y in unique_years:
            year_df = df.loc[df['year'] == y].copy()

            storage_ht = []
            previous_storage_ht = V_c
            for index, row in year_df.iterrows():
                previous_storage_ht = abs(
                    previous_storage_ht + ((row['Q_out_total'] - row['Q_in']) * row['delta_t']) / 10 ** 6)
                storage_ht.append(previous_storage_ht)

            year_df['storage_ht'] = storage_ht
            if year_df['storage_ht'].min() > V_c:
                enough_water_years += 1

        P_n = (enough_water_years / len(unique_years)) * 100

        return P_n

    def is_comprehensive_regulation(
            self,
            pre_df: pd.DataFrame,
            eps: float,
            P: float,
            round_to: int = 3,
            forced_gt_10_year: bool = True,
            forced_12_months_each_year: bool = True,
            forced_elevation: bool = True
    ):
        df = pre_df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')

        if forced_gt_10_year and not self._is_greater_than_10_years(df):
            raise ValueError("Requires at least 10 years.")

        if forced_12_months_each_year and not self._is_12_months_each_year(df):
            raise ValueError("Requires 12 months in each year.")

        pre_rows = df.shape[0]
        df = self.tjwb.calculate(
            pre_df,
            round_to
        )
        if pre_rows != df.shape[0] and forced_elevation:
            raise ValueError("Elevations in the data are invalid!")

        df = self._prepare_df_for_cal_P_n(df)

        P_n = self._calculate_P_n(df, self.V_c)

        return (P_n - P) <= eps
