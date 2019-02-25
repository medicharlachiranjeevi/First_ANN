from  model import model
import os
def reload():
       model1=model()
       if 'my_model.h5' in os.listdir('.'):
        model1.load_weights('my_model.h5')
       return model1
def save(model):
    model.save_weights('my_model.h5')

