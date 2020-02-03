from keras.constraints import Constraint
from keras import backend as K

class MaxMax(Constraint):
    def __init__(self, max_value=1,axis=2):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, 0.), K.floatx())
        max_array = K.max(w,axis=self.axis)
        max_array = K.repeat_elements(K.expand_dims(max_array,self.axis),w.shape[self.axis],axis=self.axis)
        ratio = self.max_value/(max_array+K.epsilon())
        w *=ratio
        return w

    def get_config(self):
        return {'max_value': self.max_value,
                'axis': self.axis}