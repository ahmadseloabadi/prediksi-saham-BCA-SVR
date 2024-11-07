import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from itertools import cycle
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu
import plotly.graph_objs as go
import base64
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

from pandas.tseries.offsets import BDay 

# styling for app
# Set page layout and title
st.set_page_config(page_title="STOCK PREDICTION", page_icon="img/logo.jpg")


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }

    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('img/background.jpg')

def sidebar(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-size: contain;
          background
      }}
      </style>
      """,
      unsafe_allow_html=True,
      )
sidebar('img/sidebar.jpg')

st.markdown(f"""
      <style>
   
      .stMainBlockContainer, .block-container, .st-emotion-cache-13ln4jf ,.ea3mdgi5 p{{
          color: black;       }}
      .block-container{{
          background: white;       }}
      </style>
      """,
      unsafe_allow_html=True,)

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data
def load_data():
    data = yf.download("BBRI.JK", START)
    data.reset_index(inplace=True)
    return data


data=load_data()
datas=data.copy()
tanggal=data['Date'].iloc[-1].strftime("%Y-%m-%d")
durasi=len(data['Date'])
is_today = date.today().strftime("%Y-%m-%d")


closedf= data[['Date','Close']]
close_stock = closedf.copy()
del closedf['Date']
scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

# Membuat DataFrame dari data normalisasi dengan indeks tanggal
closedf_normalized_df = pd.DataFrame(closedf, index=data['Date'], columns=['Close_Normalized'])

time_step = 4
training_size=int(len(closedf)*0.8)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]

def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]  
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Membuat DataFrame dari X_train dengan indeks y yang merupakan tanggal
X_train_df = pd.DataFrame(X_train, index=data['Date'][time_step:time_step+len(X_train)])

# Mengatur nama kolom X_train_df sebagai x0 hingga panjang data
X_train_df.columns = ['Y' if i == 3 else f'X{i+1}' for i in range(len(X_train_df.columns))]

# Membuat DataFrame dari X_test dengan indeks y yang merupakan tanggal
X_test_df = pd.DataFrame(X_test, index=data['Date'][time_step+training_size:time_step+training_size+len(X_test)])

# Mengatur nama kolom X_test_df sebagai x0 hingga panjang data
X_test_df.columns = ['Y' if i == 3 else f'X{i+1}' for i in range(len(X_test_df.columns))]

def plot_data(data):
    # Membuat candlestick chart
    candlestick = go.Candlestick(x=data['Date'],
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'])

    # Membuat layout
    layout = go.Layout(title='Grafik Candlestick Saham PT Bank Rakyat Indonesia (Persero) Tbk (BBRI.JK)',
                    xaxis=dict(title='Tanggal'),
                    yaxis=dict(title='Harga'))

    # Menggabungkan data dan layout
    fig = go.Figure(data=[candlestick], layout=layout)

    # Menampilkan grafik menggunakan Streamlit
    st.plotly_chart(fig)

def plot_train_test():
    # Membuat DataFrame dari X_train dan X_test dengan indeks tanggal
    X_train_df = pd.DataFrame(X_train, index=data['Date'][time_step:time_step+len(X_train)])
    X_test_df = pd.DataFrame(X_test, index=data['Date'][time_step+training_size:time_step+training_size+len(X_test)])

    # Menambahkan kolom untuk menandai apakah data adalah train atau test
    X_train_df['Dataset'] = 'Train'
    X_test_df['Dataset'] = 'Test'

    # Menggabungkan X_train_df dan X_test_df
    combined_df = pd.concat([X_train_df, X_test_df])

    # Membuat grafik menggunakan px.line
    fig = px.line(combined_df, x=combined_df.index, y=combined_df.columns[:-1],
                title='Pembagian Data Train dan Test 80:20',
                color='Dataset',
                labels={'value': 'Nilai', 'variable': 'Variabel', 'Date': 'Tanggal'},
                range_x=[combined_df.index.min(), combined_df.index.max()])

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.write(fig)

#fungsi make_model yang memliki satu parameter yaitu kernel_option
def make_model(kernel_options):
    # Parameter ini adalah jenis fungsi kernel yang akan digunakan dalam model SVR. Nilainya diambil dari kernel_options
    model = SVR(kernel = kernel_options)
    #melatih model
    model.fit(X_train, y_train)

    return model # Setelah melatih model, fungsi ini mengembalikan model yang sudah dilatih.

def plot_train_model(kernel_options):
    # Konversi pilihan kernel menjadi list jika belum
    if not isinstance(kernel_options, list):
        kernel_options = [kernel_options]

    # Loop melalui setiap pilihan kernel dan buat plot
    for kernel_option in kernel_options:
        model = make_model(kernel_option)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = train_predict.reshape(-1,1)
        test_predict = test_predict.reshape(-1,1)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
      
        look_back=time_step
        trainPredictPlot = np.empty_like(closedf)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(closedf)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict

        combined_predict_plot = trainPredictPlot.copy()
        combined_predict_plot[len(train_predict) + (look_back * 2) + 1:len(closedf) - 1, :] = test_predict

        plotdf = pd.DataFrame({'Date': close_stock['Date'],
                            'Original Close': close_stock['Close'],
                            'Predicted Close': combined_predict_plot.reshape(1, -1)[0].tolist()},
                            )

        # Remove rows with NaN values
        plotdf = plotdf.dropna()
        st.subheader(f'Hasil pengujian dari kernel {kernel_option}')
        st.write(f'perbandingan data aktual/original dengan data prediksi pada kernel {kernel_option}')
        st.dataframe(plotdf.set_index('Date'),use_container_width=True)

        names = cycle(['Original close price','Train predicted close price','Test predicted close price'])

        plotdf = pd.DataFrame({'Date': close_stock['Date'],
                            'original_close': close_stock['Close'],
                            'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                            'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()},
                            )

        fig = px.line(plotdf,x=plotdf['Date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                            plotdf['test_predicted_close']],
                    labels={'value':'Harga Saham bank BRI','date': 'Date'})
        fig.update_layout(title_text=f'perbandingan data original dan data prediksi pada harga close dengan kernel {kernel_option}',
                        plot_bgcolor='white', font_size=15, font_color='white',legend_title_text='Close Price')
        fig.for_each_trace(lambda t:  t.update(name = next(names)))

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.write(fig)

def prediction(kernel_option,pred_days):
    model=make_model(kernel_option)
    # Pastikan kolom 'Date' dalam bentuk datetime
    data['Date'] = pd.to_datetime(data['Date'])

    # Ambil 7 hari terakhir sebelum hari ini
    end_date = pd.Timestamp.today() - BDay(1)
    start_date = end_date - BDay(time_step)
    last_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    # Membuat prediksi untuk data uji terakhir
    last_data = closedf[-time_step:].reshape(1, -1)
    predictions = []
    dates = []
    for i in range(pred_days):
        prediction = model.predict(last_data)[0]
        predictions.append(prediction)
        last_data = np.append(last_data[:, 1:], prediction).reshape(1, -1)
        next_date = data['Date'].iloc[-1] + BDay(i+1)
        dates.append(next_date)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Membuat DataFrame untuk tabel
    predicted_df = pd.DataFrame({'Date': dates, 'Predicted Close': predictions})
    
    # Mengambil data 7 hari sebelum hari ini dari DataFrame data
    last_7_days_data = data[['Date', 'Close']].iloc[-pred_days:].copy()
    last_7_days_data.rename(columns={'Close': 'Predicted Close'}, inplace=True)

    # Menggabungkan data historis dan prediksi
    combined_df = pd.concat([last_7_days_data, predicted_df], ignore_index=True)
    
    return pred_days,combined_df

def plot_predictions(kernel,pred_days,predicted_df):
    prediction=predicted_df.iloc[-pred_days:].copy()
    st.write(f'Prediksi Harga Saham untuk {pred_days} Hari ke Depan')
    st.dataframe(prediction,hide_index=True)
    prediction.set_index('Date', inplace=True)
    # Menggabungkan DataFrame harga close asli dengan harga close yang diprediksi
    merged_df = pd.concat([close_stock.set_index('Date'), prediction], axis=1)

    # Mengisi nilai yang hilang pada harga close asli dengan harga close yang diprediksi
    merged_df['Close'] = merged_df['Close'].fillna(merged_df['Predicted Close'])
    # Menampilkan grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged_df.index, y=merged_df['Close'], mode='lines', name='all Close Price'))
    fig.add_trace(go.Scatter(x=close_stock['Date'], y=close_stock['Close'], mode='lines', name='Original Close Price'))
    fig.add_trace(go.Scatter(x=prediction.index, y=prediction['Predicted Close'], mode='lines', name='Predicted Close Price'))
    fig.update_layout(title=f'Prediksi Harga Saham untuk {pred_days} Hari ke Depan kernel {kernel}', xaxis_title='Date', yaxis_title='Close Price')
    st.plotly_chart(fig)

def mape_evaluasi(kernel_options):
    model=make_model(kernel_options)
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
    mape=mean_absolute_percentage_error(original_ytest,test_predict)
    return mape

def uji_parameter_c(kernel_option):
    c_values=[0.01,0.1,1,10]
    gamma=1
    results = []
    # ulangi model SVM dengan setiap nilai C dan gamma pada rentang yang ditentukan
    for c in c_values:
        model = SVR(kernel = kernel_option,C=c)
        #melatih model
        model.fit(X_train, y_train)

        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)

        train_predict = train_predict.reshape(-1,1)
        test_predict = test_predict.reshape(-1,1)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
        original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
        mape=mean_absolute_percentage_error(original_ytest,test_predict)
        # tambahkan hasil ke dalam list
        results.append({'kernel':kernel_option,'c': c, 'gamma': gamma,'MAPE':mape})
    results=pd.DataFrame(results)
    hasil=results.sort_values(by=['c'],ascending=True)
    st.dataframe(hasil,hide_index=True)

def mape_plot_evaluasi():
    # Simpan nilai MAPE untuk setiap kernel
    mape_values = []

    # Lakukan pelatihan model pada setiap kernel dan simpan nilai MAPE
    kernels = ['rbf', 'linear']
    for kernel in kernels:
        hasil_eval=mape_evaluasi(kernel)
        mape_values.append(hasil_eval)

    # Menampilkan nilai MAPE untuk masing-masing kernel
    st.subheader("Nilai MAPE untuk masing-masing Kernel")
    for i, kernel in enumerate(kernels):
        st.write(f'Kernel: {kernel}'+f', MAPE: {mape_values[i]:.2f}%')

    # Menampilkan grafik bar untuk membandingkan nilai MAPE
    st.subheader("Grafik Perbandingan MAPE untuk masing-masing Kernel")
    chart_data = pd.DataFrame({
        'Kernel': kernels,
        'MAPE': mape_values
    })
    st.bar_chart(chart_data.set_index('Kernel'))

def main():
    with st.sidebar :
        selected_menu = option_menu('MENU ',["HOME", "PREDICTION", "EVALUASI"])
    if selected_menu == "HOME":
        st.header("HOME")
        st.subheader(f'Data Saham PT Bank Rakyat Indonesia (Persero) Tbk (BBRI.JK)')
        st.write('yahoo finance [link](https://finance.yahoo.com/quote/BBRI.JK?.tsrc=fin-srch)')
        st.write(f'Pada tanggal {START} hingga tanggal {tanggal} dengan durasi {durasi} hari')
        st.dataframe(data,use_container_width=True)
        st.write("Data Harga Penutupan")
        st.dataframe(close_stock.set_index(close_stock.columns[0]),use_container_width=True)
        st.write("Data Normalisasi Harga Penutupan")
        st.dataframe(closedf_normalized_df,use_container_width=True)
        
        st.write("tabel data train")
        st.dataframe(X_train_df,use_container_width=True)
        st.write("tabel data test")
        st.dataframe(X_test_df,use_container_width=True)
        st.subheader('Grafik Data Saham PT Bank Rakyat Indonesia (Persero) Tbk')
        plot_data(data)
        st.subheader('Grafik pembagian data train dan test dengan perbandingan 80:20')
        plot_train_test()

    if selected_menu == "PREDICTION":
        st.header("PREDICTION")
        

        tab1,tab2=st.tabs(['PREDICTION','HISTORY'])
        with tab1:
            st.subheader("Pilih Kernel")
            kernel = st.selectbox( "Pilih Kernel yang akan digunakan:",
                ('rbf', 'linear','compire'))
            days=st.number_input(label='masukan jumlah hari prediksi ',value=7,max_value=7,min_value=1)
            if st.button('predictions') :
                
                if kernel=='compire':
                    pred_days,rbf_df=prediction('rbf',days)
                    pred_days,linear_df=prediction('linear',days)

                    pred_rbf=rbf_df.copy()
                    pred_linear=linear_df.copy()

                    eval_rbf=mape_evaluasi('rbf')
                    eval_linear=mape_evaluasi('linear')
                    merged_df = pd.merge(rbf_df, linear_df, on='Date', suffixes=('_rbf', '_linear'))

                    st.subheader(f'Tampilan Harga Saham untuk {pred_days} Hari Kedepan dan {pred_days} Hari sebelum hari ini')

                    st.dataframe(merged_df)
                    plot_predictions("rbf",pred_days,rbf_df)
                    plot_predictions("linear",pred_days,linear_df)
                    st.write(f"Nilai MAPE yang didapatkan kernel rbf adalah {eval_rbf}")
                    st.write(f"Nilai MAPE yang didapatkan kernel linear adalah {eval_linear}")

                    pred_rbf['MAE']=eval_rbf
                    pred_linear['MAE']=eval_linear
                    
                    pred_rbf.to_csv(f'history/pred_{is_today}_rbf_{pred_days}.csv',index=False)
                    pred_linear.to_csv(f'history/pred_{is_today}_linear_{pred_days}.csv',index=False)

                else:
                    pred_days,predicted_df=prediction(kernel,days)
                    # Menampilkan tabel
                    st.subheader(f'Tampilan Harga Saham untuk {pred_days} Hari Kedepan dan {pred_days} Hari sebelum hari ini')
                    st.dataframe(predicted_df,hide_index=True)
                    pred=predicted_df.copy()
                    hasil_eval=mape_evaluasi(kernel)
                    plot_predictions(kernel,pred_days,predicted_df)
                    st.write(f"Nilai MAPE yang didapatkan kernel {kernel} adalah {hasil_eval}")

                    pred['MAE']=hasil_eval
                    
                    
                    pred.to_csv(f'history/pred_{is_today}_{kernel}_{pred_days}.csv',index=False)
            with tab2:
                st.subheader('menampilkan data historis prediksi')

                date = st.date_input("masukan tanggal prediksi")
                kernel = st.multiselect( "Pilih Kernel yang akan digunakan:",['rbf', 'linear'],default=['rbf', 'linear'],key='histori')
                days=st.number_input('masukan jumlah hari prediksi ',value=7,max_value=7,min_value=1,key='history days')

                if st.button('cek history'):
                    data_not_found = False
                    for k in kernel:
                        
                        try:
                            data_history=pd.read_csv(f'history/pred_{date}_{k}_{days}.csv',sep=',')

                            st.write(f'Tampilan data history untuk kernel {k} pada tanggal {date} dengan jumlah prediksi {days} hari')

                            st.dataframe(data_history,hide_index=True)
                        except:
                            data_not_found = True
                    if data_not_found:
                        st.warning('maaf data tidak ditemukan silahkan lakukan prediksi terlebih dahulu')
                

    if selected_menu == "EVALUASI":
        tab1,tab2,tab3=st.tabs(['MAPE','PREDICT','PARAMETER'])
        with tab1:
            st.subheader('MAPE')
            st.write('penjelasan mape')
            mape_plot_evaluasi()
            st.write('penjelasan grafik dan hasil mape')
        with tab2:
            kernels = st.multiselect( "Pilih Kernel yang akan digunakan:",
                ['rbf', 'linear'])
            plot_train_model(kernels)
        with tab3:
            st.write('hasil pengujian parameter C kernel linear')
            uji_parameter_c('linear')
            st.write('hasil pengujian parameter C kernel rbf')
            uji_parameter_c('rbf')
        
if __name__ == "__main__":
    main()