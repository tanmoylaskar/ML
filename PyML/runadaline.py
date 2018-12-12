execfile("tools.py")

import AdalineGD as ad
import pandas as pd
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y = df.iloc[0:100,4].values 
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values      

niter = 10
eta   = 0.01
ada1  = ad.AdalineGD(eta=eta, n_iter=niter).fit(X,y)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')
ada2 = ad.AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), \
           ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs') 
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

# Plot weights
weights = array(ada1.wlist_)
figure()
scatter(arange(niter+1),weights[:,0])#-weights[-1,0])
scatter(arange(niter+1),weights[:,1])#-weights[-1,0])
#scatter(arange(niter+1),ppn.cost_)

X_std = np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean()) / (X[:,0]).std()
X_std[:,1] = (X[:,1]-X[:,1].mean()) / (X[:,1]).std()

ada = ad.AdalineGD(n_iter = 15, eta = 0.1)
ada.fit(X_std,y)
figure()
plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

figure()
plot(range(1, len(ada2.cost_) + 1), \
     ada2.cost_, marker='o')
set_xlabel('Epochs') 
set_ylabel('Sum-squared-error')
set_title('Adaline - Learning rate 0.0001')
plt.show()


