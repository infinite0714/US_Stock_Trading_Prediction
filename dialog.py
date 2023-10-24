import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit
from googlefinance import getQuotes

class StockSelectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Daily Stock Selection")
        self.layout = QVBoxLayout()

        self.label = QLabel("Today's Stock Selection:")
        self.layout.addWidget(self.label)

        self.stock_info = QTextEdit()
        self.stock_info.setReadOnly(True)
        self.layout.addWidget(self.stock_info)

        self.button = QPushButton("Select Stocks")
        self.button.clicked.connect(self.select_stocks)
        self.layout.addWidget(self.button)

        self.setLayout(self.layout)

    def select_stocks(self):
        sp500_tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB']  # Add more tickers as needed

        selected_stocks = []
        for ticker in sp500_tickers:
            try:
                stock_info = getQuotes(ticker)
                if stock_info:
                    selected_stocks.append({
                        'Symbol': ticker,
                        'Price': stock_info[0]['LastTradePrice'],
                        'Change': stock_info[0]['Change'],
                        'ChangePercent': stock_info[0]['ChangePercent'],
                        'MarketCap': stock_info[0]['MarketCap']
                    })
            except:
                print(f"Error fetching data for {ticker}")

        # Display the selected stocks in the text area
        self.stock_info.clear()
        for stock in selected_stocks:
            self.stock_info.append(f"Symbol: {stock['Symbol']}\n"
                                   f"Price: {stock['Price']}\n"
                                   f"Change: {stock['Change']}\n"
                                   f"Change Percent: {stock['ChangePercent']}\n"
                                   f"Market Cap: {stock['MarketCap']}\n"
                                   f"-----------------------------\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    stock_selection_app = StockSelectionApp()
    stock_selection_app.show()
    sys.exit(app.exec_())