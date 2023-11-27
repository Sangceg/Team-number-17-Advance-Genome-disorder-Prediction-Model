#AGDPM MODEL
def create_model():    
    import tensorflow as tf
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
    from tensorflow.keras.models import Model

    # n_features = 25
    # Define the input shape of the genome disorder data
    input_shape = (25,)

    # Define the number of classes for the genome disorder prediction
    num_classes = 3

    # Define the input layer
    input_layer = Input(shape=input_shape) #input layer

    # Define the fully connected layers
    fc1 = Dense(100,kernel_initializer='glorot_uniform')(input_layer)
    bn1 = BatchNormalization()(fc1)
    relu1 = Activation('relu')(bn1)
    dropout1 = Dropout(0.5)(relu1)

    fc2 = Dense(100,kernel_initializer='glorot_uniform')(dropout1)
    bn2 = BatchNormalization()(fc2)
    relu2 = Activation('relu')(bn2)
    dropout2 = Dropout(0.5)(relu2)

    fc3 = Dense(100,kernel_initializer='glorot_uniform')(dropout2)
    bn3 = BatchNormalization()(fc3)
    relu3 = Activation('relu')(bn3)
    dropout3 = Dropout(0.5)(relu3)

    fc4 = Dense(100,kernel_initializer='glorot_uniform')(dropout3)
    bn4 = BatchNormalization()(fc4)
    relu4 = Activation('relu')(bn4)
    dropout4 = Dropout(0.5)(relu4)

    fc5 = Dense(100,kernel_initializer='glorot_uniform')(dropout4)
    bn5 = BatchNormalization()(fc5)
    relu5 = Activation('relu')(bn5)
    dropout5 = Dropout(0.5)(relu5)

    fc6 = Dense(100,kernel_initializer='glorot_uniform')(dropout5)
    bn6 = BatchNormalization()(fc6)
    relu6 = Activation('relu')(bn6)
    dropout6 = Dropout(0.5)(relu6)

    # fc7 = Dense(3)(dropout6)
    # Define the output layer
    output_layer = Dense(3,kernel_initializer='glorot_uniform', activation='softmax')(dropout6)


    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    model.summary()
    from tensorflow.keras.optimizers import Adam

    # Compile the model with appropriate loss function, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'],steps_per_execution=25)

    # Train the model with appropriate training data and validation data
    # validation_data=(result_kbest_val,y1_en_val)
    # model.fit(result_kbest_20, y_t1, epochs=20, batch_size=10,validation_data=(result_kbest_val,y_val1))
    return model

model = create_model()
model.fit(result_kbest_20, y_t1, epochs=200, batch_size=25,validation_data=(result_kbest_val,y_val1))
test_loss, test_acc = model.evaluate(result_kbest_val,y_val1)
print("Test loss: ", test_loss)
print("Test Accuracy: ",test_acc)
import numpy
actual_pred = []
for ele in pred:
  actual_pred.append(numpy.argmax(ele))
pred = actual_pred
import matplotlib.pyplot as plt
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y1val, pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Mito","Mutli","Single"])
cm_display.plot()
plt.show()

from sklearn.metrics import classification_report
pred = pd.DataFrame(pred, columns =[''])
print(classification_report(y1val,pred))
