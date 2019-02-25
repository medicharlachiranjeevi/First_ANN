import warnings
from  model import model
from  savewights import save
from  savewights import reload
from sklearn.preprocessing import StandardScaler
from  dataimport import data
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import os
for filename in os.listdir('train'):
 X,y_t=data('train/'+filename)
 X_train, X_test, y_train, y_test = train_test_split(X, y_t, test_size = 0.1, random_state = 42)
 sc = StandardScaler()
 X_train = sc.fit_transform(X_train)
 X_test = sc.transform(X_test)
 model=reload()
 print('reload')
 try:
  model.fit(X_train, y_train, batch_size = 16, epochs = 200,shuffle=True)
  y_pred = model.predict(X_test)
  save(model)
  print(y_pred)
 except Exception:
  print("error")
