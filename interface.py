'''
银行操作系统界面

'''

import tkinter

root = tkinter.Tk()

root.title("自动提款机界面")

root.geometry('700x600+200+200')

# colour = ["red", "yellow", "green", "blue", "pink", "brown", "purple", "black", "white"]
# for i in colour:
#     bg = i

# 标签
label = tkinter.Label(root, text='欢迎来到暴富银行系统~', fg='green', font=('华文行楷', 20))
label.place(relx=0.265, rely=0, relwidth=0.43, relheight=0.1)

# 开户(open)
btn1 = tkinter.Button(root, text="开户(open)")
btn1.place(relx=0.07, rely=0.2, relwidth=0.16, relheight=0.05)
# 查询(search)
btn2 = tkinter.Button(root, text="查询(search)")
btn2.place(relx=0.07, rely=0.3, relwidth=0.16, relheight=0.05)
# 取款(fetch)
btn3 = tkinter.Button(root, text="取款(fetch)")
btn3.place(relx=0.07, rely=0.4, relwidth=0.16, relheight=0.05)
# 存储(save)
btn4 = tkinter.Button(root, text="存储(save)")
btn4.place(relx=0.07, rely=0.5, relwidth=0.16, relheight=0.05)
# 转账(shift)
btn5 = tkinter.Button(root, text="转账(shift)")
btn5.place(relx=0.07, rely=0.6, relwidth=0.16, relheight=0.05)

# 改密(alter)
btn6 = tkinter.Button(root, text="改密(alter)")
btn6.place(relx=0.75, rely=0.2, relwidth=0.16, relheight=0.05)
# 锁定(lock)
btn7 = tkinter.Button(root, text="锁定(lock)")
btn7.place(relx=0.75, rely=0.3, relwidth=0.16, relheight=0.05)
# 解锁(unlock)
btn8 = tkinter.Button(root, text="解锁(unlock)")
btn8.place(relx=0.75, rely=0.4, relwidth=0.16, relheight=0.05)
# 补卡(reissue)
btn9 = tkinter.Button(root, text="补卡(reissue)")
btn9.place(relx=0.75, rely=0.5, relwidth=0.16, relheight=0.05)
# 销户(cancellation)
btn10 = tkinter.Button(root, text="销户(cancellation)")
btn10.place(relx=0.75, rely=0.6, relwidth=0.16, relheight=0.05)

# 退出(quit)
btn11 = tkinter.Button(root, text="退出(quit)")
btn11.place(relx=0.4, rely=0.8, relwidth=0.16, relheight=0.05)


# 确定(confirm)
btn12 = tkinter.Button(root, text="确定(confirm)")
btn12.place(relx=0.6, rely=0.8, relwidth=0.16, relheight=0.05)
# 返回主界面(return)
btn13 = tkinter.Button(root, text="返回(return)")
btn13.place(relx=0.2, rely=0.8, relwidth=0.16, relheight=0.05)

# 单行文本输出框 用于输入
entry = tkinter.Entry(root)
entry.place(relx=0.4, rely=0.7, relwidth=0.16, relheight=0.05)

# 界面中间的文本输入框，用于显示用户输入的信息
# frame1框架
frame1 = tkinter.Frame(root, bg='white', width=300, height=100)
frame1.place(relx=0.24, rely=0.2, relwidth=0.5, relheight=0.45)
# 创建滚动条
scroll = tkinter.Scrollbar(frame1)
scroll.pack(side='right',fill='y')
# 多行文本输入框 用于显示信息
text = tkinter.Text(frame1, width=40, height=4)
text.place(relx=0, rely=0.0, relwidth=0.95, relheight=1)
# 设置默认值 'end'表示插入当前光标位置。’0.0表示插入0列0行位置
text.insert('0.0', '哈喽，欢迎来到这里~')

# 关联两个组件
scroll.config(command=text.yview)   # 控制滚动条动文本动
text.config(yscrollcommand=scroll.set)  # 文本动滚动条动



# 事件

# 为退出按钮添加的事件
# 事件函数
def quitFunc(event):
    root.quit()
# 关联
btn11.bind("<Button-1>", quitFunc)

# 为开户按钮添加的事件
# 事件函数
# def openFunc(event):

# 关联




# 用于显示窗体
root.mainloop()
