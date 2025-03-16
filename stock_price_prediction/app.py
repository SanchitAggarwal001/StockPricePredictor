import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)

model = load_model('stock_dl_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_input = request.form.get('stock')
        if not stock_input:
            stock_input = 'POWERGRID.NS'  # Default stock if none is entered

        # Allow users to enter multiple stock symbols
        stock_list = [stock.strip() for stock in stock_input.split(',')]

        results = {}

        for stock in stock_list:
            try:
            
                start = dt.datetime(2000, 1, 1)
                end = dt.datetime(2024, 10, 1)

                # Download stock data
                df = yf.download(stock, start=start, end=end)

                if df.empty:
                    results[stock] = {'error': f"No data found for {stock}"}
                    continue

                # Descriptive statistics
                data_desc = df.describe()

                # Exponential Moving Averages
                ema20 = df.Close.ewm(span=20, adjust=False).mean()
                ema50 = df.Close.ewm(span=50, adjust=False).mean()
                ema100 = df.Close.ewm(span=100, adjust=False).mean()
                ema200 = df.Close.ewm(span=200, adjust=False).mean()

                # Splitting data into training and testing sets
                data_training = df[['Close']][0:int(len(df) * 0.70)]
                data_testing = df[['Close']][int(len(df) * 0.70):]

                # Scaling training data
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_training_array = scaler.fit_transform(data_training)

                # Preparing test data
                past_100_days = data_training.tail(100)
                final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

                # Transform test data using trained scaler
                input_data = scaler.transform(final_df)

                # Creating x_test and y_test
                x_test, y_test = [], []
                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i - 100:i])
                    y_test.append(input_data[i, 0])

                x_test, y_test = np.array(x_test), np.array(y_test)

                # Make predictions
                y_predicted = model.predict(x_test)

                # Inverse scaling for predictions
                scale_factor = 1 / scaler.scale_[0]
                y_predicted = y_predicted * scale_factor
                y_test = y_test * scale_factor

                # --- Generate and save plots ---
                stock_id = stock.replace(".", "_")  

                # Plot 1: Closing Price vs Time (20 & 50 Days EMA)
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                ax1.plot(df.Close, 'y', label='Closing Price')
                ax1.plot(ema20, 'g', label='EMA 20')
                ax1.plot(ema50, 'r', label='EMA 50')
                ax1.set_title(f"{stock} - Closing Price vs Time (20 & 50 Days EMA)")
                ax1.set_xlabel("Time")
                ax1.set_ylabel("Price")
                ax1.legend()
                ema_chart_path = f"static/{stock_id}_ema_20_50.png"
                fig1.savefig(ema_chart_path)
                plt.close(fig1)

                # Plot 2: Closing Price vs Time (100 & 200 Days EMA)
                fig2, ax2 = plt.subplots(figsize=(12, 6))
                ax2.plot(df.Close, 'y', label='Closing Price')
                ax2.plot(ema100, 'g', label='EMA 100')
                ax2.plot(ema200, 'r', label='EMA 200')
                ax2.set_title(f"{stock} - Closing Price vs Time (100 & 200 Days EMA)")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Price")
                ax2.legend()
                ema_chart_path_100_200 = f"static/{stock_id}_ema_100_200.png"
                fig2.savefig(ema_chart_path_100_200)
                plt.close(fig2)

                # Plot 3: Prediction vs Original Trend
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(y_test, 'g', label="Original Price", linewidth=1)
                ax3.plot(y_predicted, 'r', label="Predicted Price", linewidth=1)
                ax3.set_title(f"{stock} - Prediction vs Original Trend")
                ax3.set_xlabel("Time")
                ax3.set_ylabel("Price")
                ax3.legend()
                prediction_chart_path = f"static/{stock_id}_stock_prediction.png"
                fig3.savefig(prediction_chart_path)
                plt.close(fig3)

                # Save dataset as CSV
                csv_file_path = f"static/{stock_id}_dataset.csv"
                df.to_csv(csv_file_path)

                # Store results
                results[stock] = {
                    'plot_path_ema_20_50': ema_chart_path,
                    'plot_path_ema_100_200': ema_chart_path_100_200,
                    'plot_path_prediction': prediction_chart_path,
                    'data_desc': data_desc.to_html(classes='table table-bordered'),
                    'dataset_link': csv_file_path
                }

            except Exception as e:
                results[stock] = {'error': str(e)}

        return render_template('index.html', results=results)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(f"static/{filename}", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
