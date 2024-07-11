# Ignore dei warnings per facilitare la lettura dei log
import warnings
warnings.filterwarnings('ignore')

# Da utlizzare come breakpoint
"""
import sys
sys.exit()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Dati OHLC (timestamp, open, high, low, close, volume, trades) dal 6 ottobre 2013 al 31 dicembre 2023 per il prezzo di Bitcoin in dollari.
# I dati sono con cadenza giornaliera.
df = pd.read_csv('Progetto\BTCUSD_Daily_OHLC.csv')
print('Head:\n',df.head(), '\n')
print('Shape:\n',df.shape, '\n')
print('Info:\n',df.info(), '\n')
print('Describe:\n',df.describe(), '\n')

# Analisi esplorativa dei dati (EDA).
# Controllo di valori nulli, non ce ne sono.
print('Valori nulli:\n',df.isnull().sum(), '\n')

# Le caratteristiche di più interesse sono il timestamp e il prezzo di chiusura close.
# Grafico con timestamp e close.
plt.plot(df['timestamp'], df['close'])
plt.title('Prezzo BTC di chiusura giornaliero.', fontsize=15)
plt.ylabel('USD')
plt.xlabel('timestamp')
plt.show()

# Creazione di una nuova caratteristica timestamp-format con il timestamp espresso in formato leggibile AAAA-MM-DD.
# Eliminazione della caratteristica timestamp.
df['timestamp-format'] = pd.to_datetime(df['timestamp'], unit='s')
df.drop('timestamp', axis=1, inplace=True)
print('Head:\n',df.head(), '\n')
print('Tail:\n',df.tail(), '\n')

# Grafico con il nuovo timestamp-format e close.
fig, ax = plt.subplots()
ax.plot(df['timestamp-format'], df['close'])
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(start, end, 365))
plt.title('Prezzo BTC di chiusura giornaliero.', fontsize=15)
plt.ylabel('USD')
plt.xlabel('Giorni (timestamp-format)')
plt.show()

# Distribuzione delle caratteristiche.
features_wo_timestamp = ['open', 'high', 'low', 'close', 'volume', 'trades']
for i, col in enumerate(features_wo_timestamp):
  plt.subplot(2,3,1+i)
  sb.histplot(df[col], kde=True)
plt.show()

for i, col in enumerate(features_wo_timestamp):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
  plt.xlabel(col)
plt.show()

for i, col in enumerate(features_wo_timestamp):
  plt.subplot(2,3,i+1)
  sb.violinplot(df[col])
  plt.xlabel(col)
plt.show()

# Ingegnerizzazione delle caratteristiche.
# Creazione di 3 nuove caratteristiche: year, month, day.
features_wo_timestamp = ['open', 'high', 'low', 'close', 'volume', 'trades']
splitted = df['timestamp-format'].astype('str').str.split('-', expand=True)
df['year'] = splitted[0].astype('int')
df['month'] = splitted[1].astype('int')
df['day'] = splitted[2].astype('int')
print('Head:\n',df.head())
print('shape:\n',df.shape)

# Distribuzioni dei valori delle caratteristiche.
data_by_year = df.groupby('year').mean(numeric_only=True)
for i, col in enumerate(features_wo_timestamp):
  plt.subplot(2,3,i+1)
  plt.ylabel(features_wo_timestamp[i])
  data_by_year[col].plot.bar()
plt.show()

# Correlazioni tra le caratteristiche.
df_corr = df.drop(columns=['timestamp-format'])
print('df_corr head:\n',df_corr.head())
sb.heatmap(df_corr.corr(), annot=True)
plt.show()

# Selezione delle caratteristiche di interesse.
df_final = df[['close']]
df_final = df[['close', 'year', 'month', 'day', 'volume', 'trades']]
print('Head final:\n',df_final.head())
print('Tail final:\n',df_final.tail())
print(df_final.info())

# Divisione dei dati in train, val e test.
n_features = df_final.shape[1]
n_lookback = 60
n_forecast = 15
n_test = n_lookback + n_forecast
df_test = df_final[int(len(df_final)-n_test):]
df_wo_test = df_final[:int(len(df_final)-n_test)]
df_train = df_wo_test[:int(0.8*len(df_wo_test))]
df_val = df_wo_test[int(0.8*len(df_wo_test)):]
print('Train shape:',df_train.shape)
print('Val shape:',df_val.shape)
print('Test shape:',df_test.shape)

# Grafico di train, val e test.
plt.figure()
plt.plot(df['timestamp-format'][:int(0.8*len(df_wo_test))].astype('str'), df_train['close'], 'green', label='Train data')
plt.plot(df['timestamp-format'][int(0.8*len(df_wo_test)):int(len(df_wo_test))].astype('str'), df_val['close'], 'blue', label='Validation data')
plt.plot(df['timestamp-format'][int(len(df_final)-n_test):].astype('str'), df_test['close'], 'red', label='Test data')
plt.xticks(np.arange(0, len(df['timestamp-format']), step=500))
plt.legend()
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.show()

# Normalizzazione dei dati.
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df_train = scaler.fit_transform(df_train)
df_val = scaler.transform(df_val)
df_test = scaler.transform(df_test)

# Modello LSTM
"""
La LSTM è composta da 4 unità principali:
Sequential: è la funzione di base per inizializzare le reti neurali.
Dense: è l'ultimo strato della rete neurale, che restituisce l'output nella forma da noi specificata.
LSTM: sono gli strati principali che dobbiamo configurare e testare per ottenere il modello migliore. Valori comunemente utilizzati vanno da 2 a 6.
Dropout: serve per evitare l'overfitting, valori bassi generano overfitting, valori alti underfitting. E' un'operazione di regolarizzazione che consiste nel disattivare casualmente un numero di unità [0,1]% di input durante l'addestramento.

Ogni strato LSTM ha 3 parametri principali:
- units: è la dimensione dello spazio di output. Valori comunemente utilizzati vanno da 50 a 150.
- return_sequence: si imposta su True se si desidera restituire la sequenza completa invece di un singolo valore. Si mette a True per gli strati intermedi.
- input_shape: è la dimensione del primo strato, che deve essere uguale alla forma dei dati di input.
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Preparazione dei dati per il modello, ricordo che:
# n_lookback = 60
# n_forecast = 15
# Strutturazione dei dati in modo che il modello analizzi 60 giorni/istanze di dati per prevedere i successivi 15 giorni/istanze di dati.
x_train = []
y_train = []
for i in range(n_lookback, df_train.shape[0] - n_forecast + 1):
    x_train.append(df_train[i-n_lookback : i, :])
    y_train.append(df_train[i : i+n_forecast, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
print('x_train shape:',x_train.shape)
print('y_train shape:',y_train.shape)

x_val = []
y_val = []
for i in range(n_lookback, df_val.shape[0] - n_forecast + 1):
    x_val.append(df_val[i-n_lookback : i, :])
    y_val.append(df_val[i : i+n_forecast, 0])
x_val, y_val = np.array(x_val), np.array(y_val)
print('x_val shape:',x_val.shape)
print('y_val shape:',y_val.shape)

# Per i dati di test, che sono 75 giorni, vengono presi i primi 60 giorni per la previsione dei successivi 15 giorni.
# Quindi le prestazioni del modello saranno calcolate sulla correttezza della previsione degli utlimi 15 giorni.
x_test = []
y_test = []
x_test.append(df_test[: n_lookback, :])
y_test.append(df_test[n_lookback : , 0])
x_test, y_test = np.array(x_test), np.array(y_test)
print('x_test shape:',x_test.shape)
print('y_test shape:',y_test.shape)
print('n_features:', n_features)


# Inizializzazione del modello
n_units = 60
n_dropout = 0.3
model = Sequential()
# LSTM layer 1
model.add(LSTM(units = n_units, return_sequences = True, input_shape = (n_lookback, n_features)))
model.add(Dropout(n_dropout))
# LSTM layer 2
model.add(LSTM(units = n_units))
model.add(Dropout(n_dropout))
# final layer
model.add(Dense(units = n_forecast))
print('Model summary:\n',model.summary())
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit del modello con strategia di uscita
"""
Parametri del fit:
epochs: numero di iterazioni. Ogni iterazione ripassa tutto il dataset. Valori comunemente utilizzati vanno da 50 a 300. Troppe epoche possono causare overfitting.
batch_size: numero di campioni da passare prima di aggiornare i pesi. Valori comunemente utilizzati vanno da 32 a 128. Un batch size troppo piccolo rallenta il processo di addestramento.
"""
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience= 10, restore_best_weights=True)
fit = model.fit(x_train, y_train, epochs = 100, batch_size = 1024, validation_data=(x_val, y_val), callbacks=[early_stop])
loss = fit.history['loss']
val_loss = fit.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.subplot(2, 1, 2)
ax = plt.gca()
ax.set_ylim([0, 0.01])
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss zoomed in")
plt.legend()
plt.show()

# Predizione
y_test_pred = model.predict(x_test)

plt.figure()
plt.plot(np.arange(0, n_lookback), x_test[0, :, 0], color = 'blue', label = 'Real Bitcoin Price')
plt.plot(np.arange(n_lookback, n_test), y_test[0, :], color = 'green', label = 'Real Bitcoin Price')
plt.plot(np.arange(n_lookback, n_test), y_test_pred[0, :], color = 'red', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction using RNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Valutazione
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test[0, :], y_test_pred[0, :])
print('MAPE: '+str(mape))

