# stratified k-fold cross validation evaluation of xgboost model
from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance, plot_tree
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
# load data
dataset = loadtxt('./coinbaseBTCUSD-withsignals-4hr-cleaned.csv', delimiter=",", skiprows=1)
# split data into X and y
X = dataset[:,0:17]
Y = dataset[:,17]
# CV model
# note that scikit will automatically determine this is a multiclass problem
model = XGBClassifier()
kfold = StratifiedKFold(n_splits=5, random_state=7)
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

model.fit(X, Y)
# plot a single decision tree
plot_tree(model, num_trees=0, rankdir='LR')
# plot feature importance
plot_importance(model)
pyplot.show()

