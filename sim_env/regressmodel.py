from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential, layers,losses, utils
import warnings

warnings.filterwarnings('ignore')


def create_X_Y(X, Y, test_rate, random_state=5):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_rate, random_state=random_state)
    return X_train, X_test, y_train, y_test


class RGModel:

    def __init__(
            self,
            X,
            Y,
            n_ft,
            batch,
            epochs,
            Xval=None,
            Yval=None,

            file_path='rgmodel.hdf5'
    ):
        self.model = Sequential([
            # 第一层
            layers.Dense(units=64,
                         activation='relu',
                         input_shape=[n_ft]),  # 输入特征11维

            # 第二层
            layers.Dense(units=64,
                         activation='relu'),

            # 第三层
            layers.Dense(units=64,
                         activation='relu'),

            layers.Dense(1)
        ])

        self.batch = batch
        self.epochs = epochs
        self.Xval = Xval
        self.Yval = Yval
        self.X = X
        self.Y = Y
        self.file_path = file_path

    def train(self):
        # Getting the untrained model
        empty_model = self.model

        # Compiling the model
        empty_model.compile(loss='mae', optimizer='adam', metrics=['mae'])

        if (self.Xval is not None) & (self.Yval is not None):
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                validation_data=(self.Xval, self.Yval),
                verbose=1
            )
        else:
            history = empty_model.fit(
                self.X,
                self.Y,
                epochs=self.epochs,
                batch_size=self.batch,
                verbose=1
            )

        # Saving to original model attribute in the class
        self.model = empty_model
        empty_model.save(filepath=self.file_path)
        # Returning the training history
        return history

    def loadmodel(self):
        self.model = load_model(self.file_path)
        return

    def predict(self, X):
        return self.model.predict(X)
