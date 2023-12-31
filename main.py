# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
import keras
import datetime
import pandas as pd
import pandas_datareader.data as web
import numpy as np

class Ui_DailyStockTrading(object):
    def setupUi(self, DailyStockTrading):
        self.flagStartbtn = False
        self.flagInvest = False
        self.InvestMoney = 0
        self.Balance = 0
        self.model = keras.models.load_model("./models/model_1")
        [self.companies, self.symbols] = self.getCompanylist(datetime.datetime.now())

        DailyStockTrading.setObjectName("Daily Stock Trading Bot")
        DailyStockTrading.resize(768, 219)
        
        # Original Setting of Invest Label
        self.lbInvest = QtWidgets.QLabel(DailyStockTrading)
        self.lbInvest.setGeometry(QtCore.QRect(10, 30, 91, 31))
        self.lbInvest.setObjectName("lbInvest")
        
        # Original Setting of Balance Label
        self.lblBalance = QtWidgets.QLabel(DailyStockTrading)
        self.lblBalance.setGeometry(QtCore.QRect(280, 29, 91, 31))
        self.lblBalance.setObjectName("lblBalance")
        # Original Setting of Invest Setting Button
        self.btnInvestSet = QtWidgets.QPushButton(DailyStockTrading)
        self.btnInvestSet.setGeometry(QtCore.QRect(577, 23, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnInvestSet.setFont(font)
        self.btnInvestSet.setObjectName("btnInvestSet")
        
        # Original Setting of Balance Log Graph Show Button
        self.btnGraphShow = QtWidgets.QPushButton(DailyStockTrading)
        self.btnGraphShow.setGeometry(QtCore.QRect(580, 160, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnGraphShow.setFont(font)
        self.btnGraphShow.setObjectName("btnGraphShow")
        
        # Original Setting of Comment of Bot
        self.lblComment = QtWidgets.QLabel(DailyStockTrading)
        self.lblComment.setGeometry(QtCore.QRect(10, 100, 741, 31))
        self.lblComment.setObjectName("lbComment")
        
        # Original Setting of Start Button
        self.btnStart = QtWidgets.QPushButton(DailyStockTrading)
        self.btnStart.setGeometry(QtCore.QRect(380, 160, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnStart.setFont(font)
        self.btnStart.setObjectName("btnStart")
        
        # Original Setting of Invest Input Control
        self.ldtInvest = QtWidgets.QLineEdit(DailyStockTrading)
        self.ldtInvest.setGeometry(QtCore.QRect(80, 24, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.ldtInvest.setFont(font)
        self.ldtInvest.setReadOnly(True)
        self.ldtInvest.setObjectName("ldtInvest")

        # Original Setting of Balance Show Control
        self.ldtBalance = QtWidgets.QLineEdit(DailyStockTrading)
        self.ldtBalance.setGeometry(QtCore.QRect(369, 24, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.ldtBalance.setFont(font)
        self.ldtBalance.setReadOnly(True)
        self.ldtBalance.setObjectName("ldtBalance")

        # translate & slot
        self.retranslateUi(DailyStockTrading)
        self.btnStart.clicked.connect(self.btnStartclicked)
        self.btnInvestSet.clicked.connect(self.btnInvestclicked)
        QtCore.QMetaObject.connectSlotsByName(DailyStockTrading)
    # Button Clicked Events
    def btnStartclicked(self):
        # timer = QTimer()
        # timer.setInterval()
        
        _translate = QtCore.QCoreApplication.translate
        if not self.flagStartbtn:
            self.btnStart.setText(_translate("DailyStockTrading", "Stop"))
            self.lblComment.setText(_translate("DailyStockTrading", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Preparing...</span></p></body></html>"))
            self.flagStartbtn = True
            today = datetime.datetime.now()
            StatusComment = 'Today is ' + str(today.strftime("%b-%d-%Y")) + '.' + " I am working to predict today's result."
            self.lblComment.setText(_translate("DailyStockTrading", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">"+ StatusComment + "</span></p></body></html>"))
            data = self.getData(today - datetime.timedelta(days = 1), 30)
            Selection = self.companies(np.argmax(self.model.predict(np.array([data]))[0]))
            StatusComment = 'Today is ' + str(today.strftime("%b-%d-%Y")) + '.<br/>' + "Result: " + Selection
            self.lblComment.setText(_translate("DailyStockTrading", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">"+ StatusComment + "</span></p></body></html>"))
            return
        else:
            self.btnStart.setText(_translate("DailyStockTrading", "Start"))
            self.lblComment.setText(_translate("DailyStockTrading", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Stopped! Press Start Button to start again</span></p></body></html>"))
            self.flagStartbtn = False
            return
    
    def btnInvestclicked(self):
        _translate = QtCore.QCoreApplication.translate
        if not self.flagInvest:
            self.ldtInvest.setReadOnly(False)
            self.btnInvestSet.setText(_translate("DailyStockTrading", "OK"))
            self.flagInvest = True
            return
        else:
            self.ldtInvest.setReadOnly(True)
            self.InvestMoney = int(self.ldtInvest.text())
            self.Balance = self.InvestMoney
            self.ldtBalance.setText(_translate("DailyStockTrading", str(self.Balance)))
            self.btnInvestSet.setText(_translate("DailyStockTrading", "Investment Setting"))
            self.flagInvest = False
            return
    # Machine Learning Part
    def getData(self, yesterday, num_days_per_company):
        start_date = yesterday - datetime.timedelta(days = num_days_per_company + 1)
        ticker_dict = {}
        for idx, symbol in enumerate(self.symbols):
            try:
                df_ticker = web.DataReader(symbol, 'iex', start=start_date, end=yesterday, api_key='pk_e11fa07e62fb48cbbe64aff1e0da7570')
                ticker_dict[symbol] = df_ticker['close']
            except:
                print(1)
                pass
        stocks = pd.DataFrame(ticker_dict)
        x = []
        for j in range(len(self.companies)):
            x.extend(list(stocks[symbol][0:]))
        return x
    #Other Functions
    def getCompanylist(self, today):
        fifteen_years_ago = today - datetime.timedelta(days=15*365)
        symbols = []
        names = []
        table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        for i in range(len(df)):
            if df.loc[i, "Symbol"] == "BF.B": # finance can't find data for BF.B, symbol may be delisted
                continue
            date_first_added = df.loc[i, "Date added"]
            if not isinstance(date_first_added, str):
                continue
            date_first_added_len = len(date_first_added)
            if date_first_added_len < 7 and date_first_added_len > 1:
                if date_first_added[date_first_added_len - 1] == "?":
                    date_first_added = date_first_added[:date_first_added_len - 1] + "-01-01"
                else:
                    date_first_added = date_first_added + "-01-01"
            elif date_first_added_len >= 7 and date_first_added_len < 10:
                if date_first_added[date_first_added_len - 1] == "?":
                    date_first_added = date_first_added[:date_first_added_len - 1] + "-01"
                else:
                    date_first_added = date_first_added + "-01"
            date_first_added = date_first_added[:10]
            date_first_added = datetime.datetime.strptime(date_first_added, "%Y-%m-%d")
            if date_first_added > fifteen_years_ago:
                continue
            symbols.append(df.loc[i, "Symbol"])
            names.append(df.loc[i, "Security"])
        return [names, symbols]
    
    # translateUi Part
    def retranslateUi(self, DailyStockTrading):
        _translate = QtCore.QCoreApplication.translate
        DailyStockTrading.setWindowTitle(_translate("DailyStockTrading", "DailyStockTradingBot"))
        self.lbInvest.setText(_translate("DailyStockTrading", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Invest</span></p></body></html>"))
        self.lblBalance.setText(_translate("DailyStockTrading", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Balance</span></p></body></html>"))
        self.btnInvestSet.setText(_translate("DailyStockTrading", "Investment Setting"))
        self.btnGraphShow.setText(_translate("DailyStockTrading", "BalanceLogGraph"))
        self.lblComment.setText(_translate("DailyStockTrading", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Welcome! I am a stock trading bot in S&amp;P 500 Companies.</span></p></body></html>"))
        self.btnStart.setText(_translate("DailyStockTrading", "Start"))
        self.ldtInvest.setText(_translate("DailyStockTrading", "0"))
        self.ldtBalance.setText(_translate("DailyStockTrading", "0"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DailyStockTrading = QtWidgets.QDialog()
    ui = Ui_DailyStockTrading()
    ui.setupUi(DailyStockTrading)
    DailyStockTrading.show()
    sys.exit(app.exec_())
