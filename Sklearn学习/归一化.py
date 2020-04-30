from sklearn import preprocessing
from scipy.stats import rankdata


x = [[1], [3], [34], [21], [10], [12]]

std_x = preprocessing.StandardScaler().fit_transform(x)

norm_x = preprocessing.MinMaxScaler().fit_transform(x)

print(rankdata(x))
print(rankdata(std_x))
print(rankdata(norm_x))
