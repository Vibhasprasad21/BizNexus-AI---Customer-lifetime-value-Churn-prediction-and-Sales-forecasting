import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

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

    def create_lstm_model(self, input_shape):
        """
        Create an enhanced Bidirectional LSTM model
        
        :param input_shape: Shape of the input data
        :return: Compiled Keras model
        """
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, prepared_data):
        """
        Train the LSTM model
        
        :param prepared_data: Dictionary containing prepared training and testing data
        :return: Trained model
        """
        # Create and train the model
        model = self.create_lstm_model((prepared_data['time_step'], 1))
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )

        # Train model with validation data
        model.fit(
            prepared_data['X_train'], 
            prepared_data['y_train'], 
            batch_size=32, 
            epochs=100, 
            validation_data=(prepared_data['X_test'], prepared_data['y_test']), 
            callbacks=[early_stopping], 
            verbose=0
        )

        return model

    def forecast(self, prepared_data, model, future_weeks=32):
        """
        Generate sales forecast
        
        :param prepared_data: Dictionary containing prepared data
        :param model: Trained LSTM model
        :param future_weeks: Number of weeks to forecast
        :return: DataFrame with original and forecasted data
        """
        resampled_data = prepared_data['resampled_data']
        train_size = prepared_data['train_size']
        time_step = prepared_data['time_step']
        
        # Get scaled data
        data = resampled_data.values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)

        # Make predictions on training and test data
        train_predictions = model.predict(prepared_data['X_train'])
        train_predictions = self.scaler.inverse_transform(train_predictions)
        
        test_predictions = model.predict(prepared_data['X_test'])
        test_predictions = self.scaler.inverse_transform(test_predictions)

        # Prepare future predictions
        future_predictions = []
        last_known_data = scaled_data[-time_step:].tolist()
        for _ in range(future_weeks):
            current_batch = np.array(last_known_data[-time_step:]).reshape((1, time_step, 1))
            future_pred = model.predict(current_batch)[0]
            future_predictions.append(future_pred[0])
            last_known_data.append(future_pred)
        future_predictions = self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Generate future dates
        last_date = resampled_data.index[-1]
        future_dates = pd.date_range(last_date, periods=future_weeks + 1, freq='W')[1:]
        future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Forecast'])

        # Combine results into a single DataFrame
        sales_forecast = pd.concat([
            pd.DataFrame({'Date': resampled_data.index, 'Original_Sales': data.flatten()}),
            pd.DataFrame({'Date': resampled_data.index[time_step:train_size], 'Train_Predictions': train_predictions.flatten()}),
            pd.DataFrame({'Date': resampled_data.index[train_size + time_step:], 'Test_Predictions': test_predictions.flatten()}),
            future_df.reset_index().rename(columns={'index': 'Date'})
        ], axis=1)

        return sales_forecast

    def visualize_forecast(self, sales_forecast, original_color="#FF7", 
                           train_color="#FF5733", test_color="#3357FF", future_color="#75FF33"):
        """
        Create a Plotly visualization of the forecast
        
        :param sales_forecast: DataFrame containing original and forecasted data
        :param original_color: Color for original sales data
        :param train_color: Color for training predictions
        :param test_color: Color for test predictions
        :param future_color: Color for future predictions
        :return: Plotly figure
        """
        # Prepare data for plotting
        fig = go.Figure()
        
        # Add traces for different components of the forecast
        fig.add_trace(go.Scatter(
            x=sales_forecast['Date'][sales_forecast['Original_Sales'].notna()], 
            y=sales_forecast['Original_Sales'][sales_forecast['Original_Sales'].notna()], 
            mode='lines', 
            name='Original Sales', 
            line=dict(color=original_color)
        ))
        
        # Train Predictions
        train_pred_mask = sales_forecast['Train_Predictions'].notna()
        fig.add_trace(go.Scatter(
            x=sales_forecast['Date'][train_pred_mask], 
            y=sales_forecast['Train_Predictions'][train_pred_mask], 
            mode='lines', 
            name='Train Predictions', 
            line=dict(color=train_color)
        ))
        
        # Test Predictions
        test_pred_mask = sales_forecast['Test_Predictions'].notna()
        fig.add_trace(go.Scatter(
            x=sales_forecast['Date'][test_pred_mask], 
            y=sales_forecast['Test_Predictions'][test_pred_mask], 
            mode='lines', 
            name='Test Predictions', 
            line=dict(color=test_color)
        ))
        
        # Future Predictions
        future_pred_mask = sales_forecast['Forecast'].notna()
        fig.add_trace(go.Scatter(
            x=sales_forecast['Date'][future_pred_mask], 
            y=sales_forecast['Forecast'][future_pred_mask], 
            mode='lines', 
            name='Future Predictions', 
            line=dict(color=future_color)
        ))

        # Update layout
        fig.update_layout(
            title='Enhanced Sales Forecasting with Bidirectional LSTM',
            xaxis_title='Date',
            yaxis_title='Sales',
            legend_title='Legend',
            template='plotly_white'
        )
        
        return fig

    def run_full_forecast(self, time_step=10, future_weeks=32):
        """
        Run the full forecasting pipeline
        
        :param time_step: Number of past time steps to use for prediction
        :param future_weeks: Number of weeks to forecast
        :return: Tuple of sales forecast DataFrame and Plotly figure
        """
        # Prepare data
        prepared_data = self.prepare_data(time_step)
        
        # Train model
        model = self.train_model(prepared_data)
        
        # Generate forecast
        sales_forecast = self.forecast(prepared_data, model, future_weeks)
        
        # Visualize forecast
        forecast_fig = self.visualize_forecast(sales_forecast)
        
        return sales_forecast, forecast_fig
    def generate_forecast(self):
        """
        Generate sales forecast
        
        Returns:
            pd.DataFrame: Sales forecast data
        """
        try:
            # Prepare time series data
            time_series = self._prepare_data()
            
            if time_series is None:
                return None
            
            # Get forecast parameters
            forecast_horizon = self.config.get('forecast_horizon', 90)
            confidence_interval = self.config.get('confidence_interval', 95) / 100.0
            
            # Generate forecast dates
            last_date = time_series['Date'].max() if 'Date' in time_series.columns else datetime.now()
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_horizon,
                freq='D'
            )
            
            # Simple forecast model (for illustration)
            # In a real implementation, use actual forecasting methods like ARIMA, Prophet, etc.
            last_values = time_series.iloc[-30:][
                'Sales' if 'Sales' in time_series.columns else time_series.columns[1]
            ].values
            
            # Calculate trend
            trend = np.polyfit(np.arange(len(last_values)), last_values, 1)[0] / np.mean(last_values)
            
            # Generate forecasted values with trend and seasonality
            base_forecast = np.mean(last_values)
            trend_multiplier = 1 + trend * np.arange(forecast_horizon)
            seasonality = 1 + 0.1 * np.sin(np.linspace(0, 2*np.pi, forecast_horizon))
            
            forecast_values = base_forecast * trend_multiplier * seasonality
            
            # Calculate confidence interval
            std_dev = np.std(last_values) / np.mean(last_values)
            z_score = 1.96 if confidence_interval >= 0.95 else 1.65  # Approximate z-score for 95% or 90% CI
            
            lower_bound = forecast_values * (1 - z_score * std_dev)
            upper_bound = forecast_values * (1 + z_score * std_dev)
            
            # Create forecast DataFrame
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