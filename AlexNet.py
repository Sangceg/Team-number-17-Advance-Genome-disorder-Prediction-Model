#ALEXNET MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Define the model architecture
model = Sequential()

model.add(Conv1D(filters=96, kernel_size=11, strides=4, activation='relu', input_shape=(25, 1)))
model.add(MaxPooling1D(pool_size=3, strides=2))

model.add(Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding="same"))
model.add(MaxPooling1D(pool_size=1, strides=1))

model.add(Conv1D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same"))
model.add(Conv1D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same"))
model.add(Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', padding="same"))
model.add(MaxPooling1D(pool_size=1, strides=1))

model.add(Flatten())

model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'],steps_per_execution=75)
model.summary()

history = model.fit(x=result_kbest_20, y=y_t1, epochs=5
                    , batch_size=75, validation_data=(result_kbest_val, y_val1))
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
