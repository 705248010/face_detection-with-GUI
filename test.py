# -*- coding: utf-8 -*-
# ch6_7.py
from tkinter import *


def calculate():  # 执行计算并显示结果
    result = eval(equ.get())
    equ.set(equ.get() + "=" + str(result))


def show(buttonString):  # 更新显示区的计算公式
    content = equ.get()
    if content == "初始化完毕":
        content = ""
    equ.set(content + buttonString)


def backspace():  # 删除前一个字符
    equ.set(str(equ.get()[:-1]))


def clear():  # 清除显示区，放置0
    equ.set("")  # equ.set("0")


root = Tk()
root.title("计算器")
# root.geometry("220x220")
equ = StringVar()
equ.set("初始化完毕")  # 默认是显示0

# 设计显示区
label = Label(root, width=68, height=6, relief="raised", anchor=SE,
              textvariable=equ)
label.grid(row=0, column=0, columnspan=4, padx=5, pady=5)

# 清除显示区按钮
clearButton = Button(root, text="清零", fg="red", width=15, height=3, command=clear)
clearButton.grid(row=1, column=0)
# 以下是row1的其他按钮
Button(root, text="退格", fg="Green", width=15, height=3, command=backspace).grid(row=1, column=1)
Button(root, text="%", width=15, height=3, command=lambda: show("%")).grid(row=1, column=2)
Button(root, text="/", width=15, height=3, command=lambda: show("/")).grid(row=1, column=3)
# 以下是row2的其他按钮
Button(root, text="7", width=15, height=3, command=lambda: show("7")).grid(row=2, column=0)
Button(root, text="8", width=15, height=3, command=lambda: show("8")).grid(row=2, column=1)
Button(root, text="9", width=15, height=3, command=lambda: show("9")).grid(row=2, column=2)
Button(root, text="*", width=15, height=3, command=lambda: show("*")).grid(row=2, column=3)
# 以下是row3的其他按钮
Button(root, text="4", width=15, height=3, command=lambda: show("4")).grid(row=3, column=0)
Button(root, text="5", width=15, height=3, command=lambda: show("5")).grid(row=3, column=1)
Button(root, text="6", width=15, height=3, command=lambda: show("6")).grid(row=3, column=2)
Button(root, text="-", width=15, height=3, command=lambda: show("-")).grid(row=3, column=3)
# 以下是row4的其他按钮
Button(root, text="1", width=15, height=3, command=lambda: show("1")).grid(row=4, column=0)
Button(root, text="2", width=15, height=3, command=lambda: show("2")).grid(row=4, column=1)
Button(root, text="3", width=15, height=3, command=lambda: show("3")).grid(row=4, column=2)
Button(root, text="+", width=15, height=3, command=lambda: show("+")).grid(row=4, column=3)
# 以下是row5的其他按钮
Button(root, text="（", width=15, height=3, command=lambda: show("(")).grid(row=5, column=0)
Button(root, text="）", width=15, height=3, command=lambda: show("）")).grid(row=5, column=1)

# 以下是row6的其他按钮
Button(root, text="0", width=32, height=3,
       command=lambda: show("0")).grid(row=6, column=0, columnspan=2)
Button(root, text=".", width=15, height=3,
       command=lambda: show(".")).grid(row=6, column=2)
Button(root, text="=", width=15, height=3,
       command=lambda: calculate()).grid(row=6, column=3)

root.mainloop()
