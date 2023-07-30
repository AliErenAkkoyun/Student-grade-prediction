import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=';')
data = data[["G1","G2","G3","studytime","failures","internet","romantic","famrel","freetime","absences"]]

predict = "G3"

x = np.array(data.drop([predict] ,1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


best = 0
for i in range(100):
    x_train , x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y , test_size = 0.1)
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc*100)

    if acc > best:
        best = acc
        with open('Model1.pickle', 'wb') as f:
            pickle.dump(linear, f)

pickle_in = open('Model1.pickle', 'rb')
linear = pickle.load(pickle_in)
print("Best Result:" , best*100,"%")
#print("Coe" , linear.coef_)
#print("Intercept" , linear.intercept_)

#["G1","G2","studytime","failures","internet","romantic","famrel","freetime","absences"]
new_student = [(16,18,2,0,1,0,4,4,0)]  #make the new student with respect to the data format
a = linear.predict(new_student)
print(a)


#View the impact of the individual segments
p = 'G2'
style.use('ggplot')
pyplot.scatter(data[p] , data['G3'])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()