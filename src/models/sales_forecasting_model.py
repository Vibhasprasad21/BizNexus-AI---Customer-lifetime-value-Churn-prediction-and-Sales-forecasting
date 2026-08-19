import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SalesForecastingModel:
    def __init__(self, customer_data=None, transaction_data=None, clv_data=None, churn_data=None, config=None):
        """
        Initialize the sales forecasting model
        
        Args:
            customer_data (pd.DataFrame): Customer data
            transaction_data (pd.DataFrame): Transaction data
            clv_data (pd.DataFrame): Customer lifetime value data
            churn_data (pd.DataFrame): Churn prediction data
            config (dict): Configuration options
        """
        self.customer_data = customer_data
        self.transaction_data = transaction_data
        self.clv_data = clv_data
        self.churn_data = churn_data
        self.config = config or {}
        
        # Validate data
        self._validate_data()
    def _validate_data(self):
        """
        Validate the input data
        """
        self.data_valid = False
        
        # Check if we have transaction data for forecasting
        if self.transaction_data is not None and isinstance(self.transaction_data, pd.DataFrame):
            # Check for required columns (adjust based on your data schema)
            required_columns = ['Customer ID']  # Add more as needed
            
            if all(col in self.transaction_data.columns for col in required_columns):
                self.data_valid = True
                
        return self.data_valid
    
        
    def _prepare_data(self):
        """
        Prepare data for forecasting
        """
        try:
            # If available, use transaction data to create time series
            if self.transaction_data is not None:
                # Check if we have a date column (find it dynamically)
                date_columns = [col for col in self.transaction_data.columns 
                               if any(date_term in col.lower() for date_term in ['date', 'time', 'day'])]
                
                if date_columns:
                    date_column = date_columns[0]
                    
                    # Check if we have purchase amount or similar
                    amount_columns = [col for col in self.transaction_data.columns 
                                    if any(amount_term in col.lower() for amount_term in 
                                          ['amount', 'price', 'revenue', 'sales', 'value'])]
                    
                    if amount_columns:
                        amount_column = amount_columns[0]

                        # Ensure date column is datetime
                        self.transaction_data[date_column] = pd.to_datetime(
                            self.transaction_data[date_column], errors='coerce'
                        )

                        # Group by date and sum the amounts
                        time_series = self.transaction_data.groupby(
                            pd.Grouper(key=date_column, freq='D')
                        )[amount_column].sum().reset_index()

                        # Fill missing dates with zeros
                        date_range = pd.date_range(
                            start=time_series[date_column].min(),
                            end=time_series[date_column].max(),
                            freq='D'
                        )

                        full_time_series = pd.DataFrame({date_column: date_range})
                        time_series = pd.merge(
                            full_time_series, time_series, on=date_column, how='left'
                        ).fillna(0)

                        # Standardize column names so every downstream method
                        # (and _generate_synthetic_data's fallback) can rely on
                        # 'Date'/'Sales' regardless of the source column names.
                        # generate_forecast() used to look for a column
                        # literally named 'Date' to find the last historical
                        # date - since real uploads are almost never named
                        # exactly that ('Order Date', 'Purchase Date', etc.),
                        # that check always failed silently and every forecast
                        # was anchored to today's date instead of the day
                        # after the actual last transaction.
                        time_series = time_series.rename(columns={date_column: 'Date', amount_column: 'Sales'})
                        return time_series
                    
            # Fallback to generating synthetic data
            return self._generate_synthetic_data()
                
        except Exception as e:
            print(f"Error preparing forecast data: {e}")
            return self._generate_synthetic_data()
    def _generate_synthetic_data(self):
        """
        Generate synthetic time series if real data can't be used
        """
        # Generate date range
        start_date = datetime.now() - timedelta(days=365)  # 1 year of historical data
        end_date = datetime.now()
        
        # Generate date index
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Simulate sales data
        np.random.seed(42)
        base_sales = 10000  # Base daily sales
        trend = np.linspace(1, 1.2, len(date_range))  # Upward trend
        seasonality = np.sin(np.linspace(0, 4*np.pi, len(date_range))) * 0.2 + 1  # Seasonal component
        noise = np.random.normal(0, 0.1, len(date_range))
        
        sales = base_sales * trend * seasonality * (1 + noise)
        
        # Create DataFrame
        time_series = pd.DataFrame({
            'Date': date_range,
            'Sales': sales
        })
        
        return time_series
    def _apply_churn_impact(self, forecast_df):
        """
        Apply churn predictions to the forecast if available
        """
        if self.churn_data is not None and 'Churn_Probability' in self.churn_data.columns:
            # Get average churn probability
            avg_churn_prob = self.churn_data['Churn_Probability'].mean()
            
            # Apply a simple adjustment to forecasted values based on churn
            forecast_reduction_factor = 1 - (avg_churn_prob * 0.5)  # Reduce impact by half
            
            # Apply reduction to the forecast
            forecast_df['Forecast'] = forecast_df['Forecast'] * forecast_reduction_factor
            
            # Adjust the bounds as well
            forecast_df['Lower_Bound'] = forecast_df['Lower_Bound'] * forecast_reduction_factor
            forecast_df['Upper_Bound'] = forecast_df['Upper_Bound'] * forecast_reduction_factor
            
            return forecast_df
        
        return forecast_df

    def generate_forecast(self):
        """
        Generate a daily sales forecast, driven by a monthly trend/seasonality
        fit rather than raw daily values.

        Daily retail transaction totals are dominated by day-to-day noise
        (weekends, one large order, stockouts) - fitting a trend line or a
        seasonal pattern directly on the last 30 raw daily values, as an
        earlier version of this method did, means the forecast mostly
        reflects whichever few days happened to land in that window. On real
        data that produced a 90-day-ahead forecast trend of roughly +200%
        (confirmed against Superstore, where actual month-to-month growth is
        mild) and a fixed-frequency sine wave standing in for "seasonality"
        that has no relationship to the dataset's actual calendar pattern.

        This aggregates to monthly totals first, fits the trend on that (far
        less noisy) series, and - when there's enough history - derives a
        real per-calendar-month seasonal index via ratio-to-moving-average
        decomposition instead of an arbitrary sine wave. The result is
        distributed back out to daily rows since the rest of the app expects
        one row per day.

        Returns:
            pd.DataFrame: Sales forecast data (Date, Forecast, Lower_Bound, Upper_Bound)
        """
        try:
            # Prepare time series data
            time_series = self._prepare_data()

            if time_series is None or time_series.empty:
                return None

            forecast_horizon = self.config.get('forecast_horizon', 90)
            confidence_interval = self.config.get('confidence_interval', 95) / 100.0
            last_date = time_series['Date'].max()

            monthly = time_series.set_index('Date')['Sales'].resample('MS').sum()

            # Drop a trailing month that's materially incomplete (data ends
            # mid-month) so it doesn't bias the trend/seasonality fit low.
            month_end = last_date + pd.offsets.MonthEnd(0)
            if (month_end - last_date).days > 3 and len(monthly) > 1:
                monthly = monthly.iloc[:-1]

            if len(monthly) == 0:
                return None

            # --- Trend: fit on up to the last 24 months, capped to a sane
            # per-month rate so a short or volatile history can't compound
            # into an absurd multi-month extrapolation. ---
            trend_window = monthly.tail(24)
            if len(trend_window) >= 2 and trend_window.mean() > 0:
                x = np.arange(len(trend_window))
                slope, intercept = np.polyfit(x, trend_window.values, 1)
                monthly_trend_rate = slope / trend_window.mean()
            else:
                intercept = trend_window.mean() if len(trend_window) else 0.0
                monthly_trend_rate = 0.0
            monthly_trend_rate = float(np.clip(monthly_trend_rate, -0.15, 0.15))
            trend_line_last = max(intercept + (slope if len(trend_window) >= 2 else 0) * (len(trend_window) - 1), 0) \
                if len(trend_window) >= 1 else 0.0

            # --- Seasonality: a real per-calendar-month index derived from
            # the data's own history (ratio-to-moving-average), only when
            # there's enough history (>=12 months) to estimate one reliably.
            # Otherwise assume no seasonality rather than fabricate a pattern. ---
            seasonal_index = pd.Series(1.0, index=range(1, 13))
            if len(monthly) >= 12:
                rolling_mean = monthly.rolling(window=12, center=True, min_periods=6).mean()
                ratio = (monthly / rolling_mean.replace(0, np.nan)).dropna()
                if not ratio.empty:
                    by_month = ratio.groupby(ratio.index.month).mean()
                    by_month = by_month.reindex(range(1, 13)).fillna(1.0)
                    # Normalize so the twelve indices average to 1.0
                    if by_month.mean() > 0:
                        by_month = by_month / by_month.mean()
                    seasonal_index = by_month

            # --- Project forward month by month, then spread each month's
            # total evenly across its days to build the daily output rows. ---
            n_forecast_months = int(np.ceil(forecast_horizon / 30.44)) + 1
            future_months = pd.date_range(
                start=(last_date + pd.offsets.MonthBegin(1)), periods=n_forecast_months, freq='MS'
            )

            monthly_forecast = {}
            for i, month_start in enumerate(future_months, start=1):
                growth_factor = (1 + monthly_trend_rate) ** i
                season = seasonal_index.get(month_start.month, 1.0)
                monthly_forecast[month_start] = max(trend_line_last * growth_factor * season, 0)

            # Historical volatility of monthly totals around their own
            # month's seasonal-adjusted trend, used to size the confidence band.
            deseasonalized = monthly / monthly.index.map(lambda d: seasonal_index.get(d.month, 1.0))
            cv = (deseasonalized.std() / deseasonalized.mean()) if deseasonalized.mean() else 0.3
            cv = float(np.clip(cv, 0.05, 1.0))
            z_score = 1.96 if confidence_interval >= 0.95 else 1.65

            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon, freq='D')
            daily_rows = []
            for date in forecast_dates:
                month_start = date.replace(day=1)
                days_in_month = pd.Period(date, freq='M').days_in_month
                month_total = monthly_forecast.get(month_start)
                if month_total is None:
                    # Horizon ran past the months we projected - hold the last one flat.
                    month_total = list(monthly_forecast.values())[-1] if monthly_forecast else 0
                daily_value = month_total / days_in_month
                daily_rows.append(daily_value)

            forecast_values = np.array(daily_rows)
            lower_bound = forecast_values * (1 - z_score * cv)
            upper_bound = forecast_values * (1 + z_score * cv)

            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast_values,
                'Lower_Bound': lower_bound,
                'Upper_Bound': upper_bound
            })

            # Apply churn impact if available
            forecast_df = self._apply_churn_impact(forecast_df)

            return forecast_df

        except Exception as e:
            print(f"Error generating forecast: {e}")
            return None