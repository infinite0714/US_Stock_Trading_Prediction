# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dialog.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QDialog, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QWidget)

class Ui_DailyStockTrading(object):
    def setupUi(self, DailyStockTrading):
        if not DailyStockTrading.objectName():
            DailyStockTrading.setObjectName(u"DailyStockTrading")
        DailyStockTrading.resize(768, 219)
        self.lbInvest = QLabel(DailyStockTrading)
        self.lbInvest.setObjectName(u"lbInvest")
        self.lbInvest.setGeometry(QRect(10, 30, 91, 31))
        self.lblBalance = QLabel(DailyStockTrading)
        self.lblBalance.setObjectName(u"lblBalance")
        self.lblBalance.setGeometry(QRect(280, 29, 91, 31))
        self.btnInvestSet = QPushButton(DailyStockTrading)
        self.btnInvestSet.setObjectName(u"btnInvestSet")
        self.btnInvestSet.setGeometry(QRect(577, 23, 171, 41))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.btnInvestSet.setFont(font)
        self.btnGraphShow = QPushButton(DailyStockTrading)
        self.btnGraphShow.setObjectName(u"btnGraphShow")
        self.btnGraphShow.setGeometry(QRect(580, 160, 171, 41))
        self.btnGraphShow.setFont(font)
        self.lbComment = QLabel(DailyStockTrading)
        self.lbComment.setObjectName(u"lbComment")
        self.lbComment.setGeometry(QRect(10, 100, 741, 31))
        self.btnStart = QPushButton(DailyStockTrading)
        self.btnStart.setObjectName(u"btnStart")
        self.btnStart.setGeometry(QRect(380, 160, 171, 41))
        self.btnStart.setFont(font)
        self.ldtInvest = QLineEdit(DailyStockTrading)
        self.ldtInvest.setObjectName(u"ldtInvest")
        self.ldtInvest.setGeometry(QRect(80, 24, 191, 41))
        font1 = QFont()
        font1.setPointSize(14)
        font1.setBold(True)
        self.ldtInvest.setFont(font1)
        self.ldtBalance = QLineEdit(DailyStockTrading)
        self.ldtBalance.setObjectName(u"ldtBalance")
        self.ldtBalance.setGeometry(QRect(369, 24, 191, 41))
        self.ldtBalance.setFont(font1)
        self.ldtBalance.setReadOnly(True)

        self.retranslateUi(DailyStockTrading)
        self.btnStart.clicked.connect(self.lbComment.clear)

        QMetaObject.connectSlotsByName(DailyStockTrading)
    # setupUi

    def retranslateUi(self, DailyStockTrading):
        DailyStockTrading.setWindowTitle(QCoreApplication.translate("DailyStockTrading", u"DailyStockTradingBot", None))
        self.lbInvest.setText(QCoreApplication.translate("DailyStockTrading", u"<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Invest</span></p></body></html>", None))
        self.lblBalance.setText(QCoreApplication.translate("DailyStockTrading", u"<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Balance</span></p></body></html>", None))
        self.btnInvestSet.setText(QCoreApplication.translate("DailyStockTrading", u"Investment Setting", None))
        self.btnGraphShow.setText(QCoreApplication.translate("DailyStockTrading", u"BalanceLogGraph", None))
        self.lbComment.setText(QCoreApplication.translate("DailyStockTrading", u"<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:600;\">Welcome! I am a stock trading bot in S&amp;P 500 Companies.</span></p></body></html>", None))
        self.btnStart.setText(QCoreApplication.translate("DailyStockTrading", u"Start", None))
        self.ldtInvest.setInputMask("")
        self.ldtInvest.setText(QCoreApplication.translate("DailyStockTrading", u"0", None))
        self.ldtBalance.setText(QCoreApplication.translate("DailyStockTrading", u"0", None))
    # retranslateUi

