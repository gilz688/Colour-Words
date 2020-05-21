from tkinter import *
from tkinter.colorchooser import *
from tkinter import messagebox

from colour_models import svmModel
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

def train():
    data = pd.read_csv('colour-data.csv')
    X = data[['R', 'G', 'B']].values / 255
    y = data['Label'].values
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=8)

    model = svmModel()
    model.fit(X_train, y_train)

    return model

model = train()

def getColor():
    color = askcolor() 
    rgb = np.array(color[0])/255
    y = model.predict(rgb)
    messagebox.showinfo("Color", "You picked \"" + y[0] + "\".")

def main():
    root = Tk()
    root.title = "Color Words"
    button = Button(root, text='Select Color', command=getColor)
    button.pack()
    root.mainloop()

if __name__ == '__main__':
    main()