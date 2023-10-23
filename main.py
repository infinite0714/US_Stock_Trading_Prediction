# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(768, 684)
        self.Investment = QtWidgets.QLabel(Dialog)
        self.Investment.setGeometry(QtCore.QRect(11, 30, 91, 31))
        self.Investment.setObjectName("Investment")
        self.tbxInvested_Money = QtWidgets.QLineEdit(Dialog)
        self.tbxInvested_Money.setGeometry(QtCore.QRect(108, 24, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.tbxInvested_Money.setFont(font)
        self.tbxInvested_Money.setObjectName("tbxInvested_Money")
        self.Total = QtWidgets.QLabel(Dialog)
        self.Total.setGeometry(QtCore.QRect(302, 29, 61, 31))
        self.Total.setObjectName("Total")
        self.tbxTotal_Money = QtWidgets.QLineEdit(Dialog)
        self.tbxTotal_Money.setGeometry(QtCore.QRect(355, 24, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.tbxTotal_Money.setFont(font)
        self.tbxTotal_Money.setObjectName("tbxTotal_Money")
        self.btnAdd_Invested = QtWidgets.QPushButton(Dialog)
        self.btnAdd_Invested.setGeometry(QtCore.QRect(577, 23, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnAdd_Invested.setFont(font)
        self.btnAdd_Invested.setObjectName("btnAdd_Invested")
        self.From_date = QtWidgets.QDateEdit(Dialog)
        self.From_date.setGeometry(QtCore.QRect(110, 90, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.From_date.setFont(font)
        self.From_date.setObjectName("From_date")
        self.From = QtWidgets.QLabel(Dialog)
        self.From.setGeometry(QtCore.QRect(13, 95, 91, 31))
        self.From.setObjectName("From")
        self.To = QtWidgets.QLabel(Dialog)
        self.To.setGeometry(QtCore.QRect(299, 94, 51, 31))
        self.To.setObjectName("To")
        self.To_date = QtWidgets.QDateEdit(Dialog)
        self.To_date.setGeometry(QtCore.QRect(356, 89, 181, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.To_date.setFont(font)
        self.To_date.setObjectName("To_date")
        self.ListctrlBot_Action_Log = QtWidgets.QListView(Dialog)
        self.ListctrlBot_Action_Log.setGeometry(QtCore.QRect(10, 140, 531, 541))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(True)
        self.ListctrlBot_Action_Log.setFont(font)
        self.ListctrlBot_Action_Log.setObjectName("ListctrlBot_Action_Log")
        self.btnStart = QtWidgets.QPushButton(Dialog)
        self.btnStart.setGeometry(QtCore.QRect(580, 150, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnStart.setFont(font)
        self.btnStart.setObjectName("btnStart")
        self.btnStop = QtWidgets.QPushButton(Dialog)
        self.btnStop.setGeometry(QtCore.QRect(580, 240, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnStop.setFont(font)
        self.btnStop.setObjectName("btnStop")
        self.btnGraphShow = QtWidgets.QPushButton(Dialog)
        self.btnGraphShow.setGeometry(QtCore.QRect(580, 338, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnGraphShow.setFont(font)
        self.btnGraphShow.setObjectName("btnGraphShow")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.Investment.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Invested</span></p></body></html>"))
        self.Total.setText(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">Total</span></p></body></html>"))
        self.btnAdd_Invested.setText(_translate("Dialog", "Add Investment"))
        self.From.setText(_translate("Dialog", "<html><head/><body><p align=\"right\"><span style=\" font-size:12pt; font-weight:600;\">From</span></p></body></html>"))
        self.To.setText(_translate("Dialog", "<html><head/><body><p align=\"right\"><span style=\" font-size:12pt; font-weight:600;\">To</span></p></body></html>"))
        self.btnStart.setText(_translate("Dialog", "Start"))
        self.btnStop.setText(_translate("Dialog", "Stop"))
        self.btnGraphShow.setText(_translate("Dialog", "Graph Show"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())