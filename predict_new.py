from  savewights import reload
import warnings

from  dataimport import data_pred
warnings.filterwarnings("ignore")
X=data_pred('pre.csv')
model=reload()
y_pred = model.predict(X)
print(y_pred)

