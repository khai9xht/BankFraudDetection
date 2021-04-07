import sys
sys.path.append("/home/hoangnv68/BankFraudDetection")
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import keras_metrics
from utils import read_data, Downsample_data, convert_data
from model import build_model
import os


def train(model, X_train, X_test, y_train, y_test, save_path):
    model.compile(Adam(lr=0.001), loss="sparse_categorical_crossentropy", \
            metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    model.fit(X_train, y_train, \
            validation_data=(X_test, y_test), \
            batch_size=32, \
            epochs=20
        )
    model.save(save_path)
    # return model


if __name__ == "__main__":
    print("-"*80)
    print("Start training Neurol Network model...")
    path = "/home/hoangnv68/BankFraudDetection/creditcard.csv"
    df = read_data(path)
    sub_df = Downsample_data(df)
    X = sub_df.drop("Class", axis=1)
    y = sub_df["Class"]

    X_train, X_test, y_train, y_test = convert_data(X, y)

    model = build_model(X_train.shape[1], 2)
    print("model structure:\n", model.summary())
    save_path = "pretrained_model"
    name_model = "Exmodel.h5"
    model_path = os.path.join(save_path, name_model)
    train(model, X_train, X_test, y_train, y_test, model_path)
    print("-"*80)
    print("Saving model...")
    print("save model successully !!!")
    print("-"*80)