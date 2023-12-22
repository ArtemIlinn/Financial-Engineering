import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
pd.options.mode.chained_assignment = None


"""
best result if not short [100000.02134224417, 1, 25, 66]
15m intervals
buy_threshold = 25
sell_threshold = 66
"""


class RSI_strategy:
    def __init__(self, 
                    data, 
                    window=14,
                    initial_budget = 1000,
                    buy_threshold = 30,
                    sell_threshold = 70,
                    leverage=0
                 ):
        """
        :param df: the timeseies data, looking for "Adj Close" column
        :param window: window size
        :param l: lower threshold
        :param u: upper threshold
        """
        
        self.data = data
        self.window = window

        self.initial_budget = initial_budget
        self.capital = initial_budget

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.average_gain = None
        self.average_loss = None

        self.asset_quantity = 0
        self.leverage = leverage


    def get_signals(self):
        """
        Calcultaes prices' Difference, Upmoves, Downmoves, RS, RSI and adds respective columns to the dataframe
        defines the position, 0: none, -1: sell/short, 1: buy
        """

        close_price = self.data['Adj Close']
        daily_return = close_price.diff()
        gain = daily_return.where(daily_return > 0, 0)
        loss = -daily_return.where(daily_return < 0, 0)

        avg_gain = gain.rolling(window=self.window, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        self.data['RSI'] = rsi

        # Generate trading signals based on RSI values
        self.data['Position'] = 0  # 0 represents no action
        self.data['Position'][self.data['RSI'] < self.buy_threshold] = 1  # Buy signal
        self.data['Position'][self.data['RSI'] > self.sell_threshold] = -1  # Sell signal

        # Carry forward the position until the opposite signal is generated
        self.data['Position'] = self.data['Position'].fillna(method='ffill')

        # Replace all zeros in Position with the previous non-zero position
        self.data['Position'] = self.data['Position'].replace(0, method='ffill')      


        

    
    def run_rsi(self, return_history=False, short=False, leverage=0):
        """
        Runs RSI strategy
        Returns pandas dataframe - history of trades if return_history=True
        leverage=1 indicates we do not leverage any assets from the side, otherweise integer
        """
        
        history = self.data.copy()
        self.leverage = leverage

        if not short:  
            # Calculate daily PnL based on the trading signals
            history['PnL'] = history['Adj Close'].pct_change() * history['Position'].shift(1)

            # Calculate cumulative PnL and plot the results
            history['Cumulative PnL'] = history['PnL'].cumsum()

            # Calculate wealth based on an initial budget of $1000
            history['Wealth'] = self.initial_budget + history['PnL'].cumsum()


        try:
            final_cumulative_profit = history['Wealth'].iloc(-1) 
        except:
            final_cumulative_profit = 0

            
        if return_history:
            return final_cumulative_profit, history
        else:
            return final_cumulative_profit


    

    def calibrate(self,  short=False, leverage=0):
        max_profit = self.initial_budget
        brute_force_history = []  # lists like [current_profit, window, buy, sell]

        #ts = self.data.copy()

        for window in [14, 21, 30]: # 97 in this case 96-th will get us period of 24 hours
            for buy in [20, 25, 30, 35]: # 20, 51
                for sell in [55, 65, 70, 75]: # 50, 81


                    self.window = window
                    self.buy_threshold = buy
                    self.sell_threshold = sell

                    self.capital = self.run_rsi(short=False, leverage=0)
                    brute_force_history.append([self.capital, window, buy, sell])

        return brute_force_history
    

    def plot(self, plot_signals=True):

        """
        Plots Adj Close along with RSI
        """

        colors = [(0, 1, 1), (1, 0, 1)]  # 'cyan' to 'magenta'
        custom_colormap = LinearSegmentedColormap.from_list('custom_colormap', colors)

        plt.figure(figsize=(15, 8))

        # Plot for asset movements
        ax1 = plt.subplot(211)
        ax1.plot(self.data.index, self.data['Adj Close'], color='cyan', linewidth=2.0)
        ax1.set_title('Adj Close')
        ax1.grid(True)
        ax1.set_axisbelow(True)
        ax1.set_facecolor('black')

        # Plot for RSI
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(self.data.index, self.data['RSI'], color='blue')

        for y_val, linestyle, color in zip([0, 10, 20, 30, 50, 70, 80, 90, 100],
                                     ['-', '-.', '--', ':', '-', ':', '--', '-.', '-', '-'], 
                                     #['-', '--', ':', '-.', '-', '--', ':', '-.', '--'],
                                   custom_colormap(np.linspace(0, 1, 9))):
            ax2.axhline(y_val, linestyle=linestyle, alpha=1, color=color)

        ax2.axhline(self.buy_threshold, linestyle='-', alpha=1, color='green', label='Buy Threshold')
        ax2.axhline(self.sell_threshold, linestyle='-', alpha=1, color='red', label='Sell Threshold')

        # Coloring 
        ax2.axhspan(self.sell_threshold, 100, facecolor='magenta', alpha=0.3, label='Sell Zone')
        ax2.axhspan(self.sell_threshold, self.buy_threshold, facecolor='blue', alpha=0.3, label='Sell Zone')
        ax2.axhspan(0, self.buy_threshold, facecolor='cyan', alpha=0.3, label='Buy Zone')

        ax2.set_title("RSI")
        ax2.grid(False)
        ax2.set_axisbelow(True)
        # ax2.legend()

        if plot_signals:
            # Plotting vertical lines on ax1 based on "Position" values
            last_position = 0

            for date, position in self.data[self.data['Position'] != 0].iterrows():
                if position['Position'] == 1 and last_position != 1:
                    ax1.axvline(date, color='springgreen', linestyle='--', alpha=1, label='Buy Signal', linewidth=2.0)
                elif position['Position'] == -1 and last_position != -1:
                    ax1.axvline(date, color='red', linestyle='--', alpha=1, label='Sell Signal', linewidth=2.0)

                last_position = position['Position']

        plt.show()


    def plot_history(self, history):
        # Plot the Closing Price, RSI, Trading Signals, Cumulative PnL, and Wealth
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

        ax1.plot(history.index, history['Adj Close'], label='Close Price', color='cyan')
        ax1.set_ylabel('Adj Close Price')
        #ax1.set_facecolor('black')
        #ax1.legend()

        ax2.plot(history.index, history['RSI'], label='RSI', color='blue')
        ax2.axhline(self.buy_threshold, color='green', linestyle='--', label='Oversold Threshold')
        ax2.axhline(self.sell_threshold, color='red', linestyle='--', label='Overbought Threshold')
        ax2.set_ylabel('RSI')
        #ax2.legend()

        ax3.plot(history.index, history['Position'], label='Trading Signals', color='magenta', marker='o')
        ax3.set_ylabel('Position')
        #ax3.legend()

        ax4.plot(history.index, history['Cumulative PnL'], label='Cumulative PnL', color='orange')
        ax4.set_ylabel('Cumulative PnL')
        #ax4.legend()
        #ax4.set_facecolor('black')

        ax5.plot(history.index, history['Wealth'], label='Wealth', color='lightseagreen')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Wealth')
        #ax5.set_facecolor('black')
        #ax5.legend()

        plt.show()


    def calculate_drawdown(self, history):
        cumulative_pnl = history['PnL']
        high_watermark = cumulative_pnl.iloc[0]
        realized_drawdown = 0
        unrealized_drawdown = 0

        for pnl in cumulative_pnl:
            if pnl > high_watermark:
                high_watermark = pnl
            else:
                drawdown = high_watermark - pnl
                unrealized_drawdown = max(unrealized_drawdown, drawdown)

        realized_drawdown = cumulative_pnl.min() - cumulative_pnl.iloc[0]

        return realized_drawdown, unrealized_drawdown
    
    def profitable_signals(self, history):
        # Calculate the frequency of profitable signals
        profitable_signals = history[history['PnL'] > 0]
        frequency_profitable_signals = len(profitable_signals) / len(history)

        print(f'Frequency of Profitable Signals: {frequency_profitable_signals:.2%}')

        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(history.index, history['Adj Close'], label='Adj Close Price', color='blue')
        plt.scatter(profitable_signals.index, profitable_signals['Adj Close'], marker='^', color='green', label='Profitable Signal')
        plt.scatter(history.index[history['Position'] == -1], history['Close'][history['Position'] == -1], marker='v', color='red', label='Sell Signal')
        plt.scatter(history.index[history['Position'] == 1], history['Close'][history['Position'] == 1], marker='^', color='green', label='Buy Signal')
        plt.title('RSI Trading Signals and Profitable Signals')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()


    def testing(self, history):
        # Split the data into in-sample and out-of-sample
        split_date = '2023-12-12'
        in_sample = history.loc[history.index < split_date]
        out_of_sample = history.loc[history.index >= split_date]

        # Plot in-sample results
        plt.figure(figsize=(12, 6))
        plt.plot(in_sample.index, in_sample['Close'], label='Close Price', color='blue')
        plt.scatter(in_sample.index, in_sample['Close'][in_sample['Position'] == 1], marker='^', color='green', label='Buy Signal')
        plt.scatter(in_sample.index, in_sample['Close'][in_sample['Position'] == -1], marker='v', color='red', label='Sell Signal')
        plt.title('In-Sample Testing Results')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()

        # Plot out-of-sample results
        plt.figure(figsize=(12, 6))
        plt.plot(out_of_sample.index, out_of_sample['Close'], label='Close Price', color='blue')
        plt.scatter(out_of_sample.index, out_of_sample['Close'][out_of_sample['Position'] == 1], marker='^', color='green', label='Buy Signal')
        plt.scatter(out_of_sample.index, out_of_sample['Close'][out_of_sample['Position'] == -1], marker='v', color='red', label='Sell Signal')
        plt.title('Out-of-Sample Testing Results')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()
    
    