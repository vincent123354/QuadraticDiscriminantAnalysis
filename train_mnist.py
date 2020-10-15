import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from qda import QDA

if __name__=='__main__':
	X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
	y = y.astype(int)

	x_train, x_test, y_train, y_test = train_test_split(X, y)

	qda = QDA()
	qda.fit(x_train, y_train)
	train_acc = (qda.predict(x_train).argmax(1) == y_train.squeeze()).mean() * 100
	test_acc = (qda.predict(x_test).argmax(1) == y_test.squeeze()).mean() * 100

	print(f'Train Accuracy : {train_acc}%')
	print(f'Test Accuracy : {test_acc}%')

	# plot generated
	images = []

	for i in range(10): # each class
	    temp = []
	    for j in range(10): # 10 samples
	        temp.append(qda.generate(i).reshape(28,28))
	    images.append(temp)

	images = np.array(images)

	fig = plt.figure(figsize=(20,20))
	gs1 = gridspec.GridSpec(10, 10)
	gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

	for i in range(100):
	    ax = fig.add_subplot(gs1[i])
	    ax.imshow(images[i // 10][i %10], cmap='gray')
	    if i % 10 == 0:
	        ax.set_ylabel(str(i // 10), rotation=0, fontsize=20, labelpad=10)
	    ax.set_xticklabels([])
	    ax.set_yticklabels([])
	    ax.axes.xaxis.set_ticks([])
	    ax.axes.yaxis.set_ticks([])

	plt.show()
