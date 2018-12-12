import perceptron as PN
import pandas as pd
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
y = df.iloc[0:100,4].values 
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values      

niter = 10
ppn = PN.Perceptron(eta=0.1, n_iter=niter)
ppn.fit(X,y)

# Plot weights
weights = array(ppn.wlist_)
scatter(arange(niter+1),weights[:,0]-weights[-1,0])
scatter(arange(niter+1),weights[:,1]-weights[-1,0])
