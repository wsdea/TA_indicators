import gdax
import datetime as dt
import pandas as pd
import numpy as np
import time

def add_French_Time(row,data,start):
    #Adding Time from TimeStamp
    first_time_stamp = data.iloc[0].name
    seconds_added = int(row.name - first_time_stamp)
    row['UTC-Time'] = (start + dt.timedelta(days=0, hours=0, minutes=0, seconds = seconds_added)).isoformat()
    row['French_Time'] = (start + dt.timedelta(days=0, hours=2, minutes=0, seconds = seconds_added)).isoformat()
    return row

def get_data_from_GDAX(pair, candle_size='15M', start = dt.datetime(2015,1,1), N = 3000, max_candles_per_page = 300):
    #N is the desired number of candles
    g = gdax.PublicClient()
    candle_sizes = {'1M' : 60,'5M':300,'15M':900,'1H':3600,'6H':21600,'1D':86400}
    
    #gathering data from api
    start = start - dt.timedelta(days=0, hours=2, minutes=0) #lag of 2 hours to start at midnight french time
    deltaT = dt.timedelta(days=0, hours=0, minutes=max_candles_per_page*candle_sizes[candle_size]/60)
    
    T = [start + i*deltaT for i in range(0,N//max_candles_per_page)]
    history = []
    for t in T:
        try:
            h = g.get_product_historic_rates(pair, start = t.isoformat(), end = (t + deltaT).isoformat(), granularity=candle_sizes[candle_size])
        except:
            print("bad handshake, waiting 15 sec")
            time.sleep(15)
            h = g.get_product_historic_rates(pair, start = t.isoformat(), end = (t + deltaT).isoformat(), granularity=candle_sizes[candle_size])
        
        history += h
        time.sleep(0.4)
        print(history[-1]==history[-2]) #checking if the data doesn't repeat (it does it when gdax's api is requested too much)
    
    try:
        #processing data
        data = pd.DataFrame(history,columns = ['TimeStamp','low','high','open','close','volume']).set_index('TimeStamp')
        data = data.sort_index()
        #    print("Data contient les données du {} au {}".format(T[0].isoformat(),(T[-1]+deltaT).isoformat()))
        data = data.apply(lambda x : add_French_Time(x,data,start),axis = 1)
        return data
    except:
        print('error')
        return history


def RSI(source,n=14):
    #https://www.investopedia.com/terms/r/rsi.asp
    #same as Trading View
    delta = source.diff()
    # Get rid of the first row, which is NaN since it did not have a previous row to calculate the differences
    delta = delta[1:] 
    
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = up.ewm(com = n-1).mean()
    roll_down = down.abs().ewm(com = n-1).mean()
    
    # Calculate the RSI
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def MACD(source,fast=24,slow=52,signal=9):
    #https://www.investopedia.com/terms/m/macd.asp
    #about the same as Trading View
    fast_MA = source.ewm(span = fast).mean()
    slow_MA = source.ewm(span = slow).mean()
    MAC = fast_MA - slow_MA
    sig_MA = MAC.ewm(span = signal).mean()
    MACD = MAC - sig_MA
    return MACD

def SMA(source,n=50):
    #https://www.investopedia.com/terms/s/sma.asp
    return source.rolling(window = n).mean()

def EMA(source,n=50):
    #https://www.investopedia.com/terms/e/ema.asp
    return source.ewm(span = n).mean()

def BollingerBands(source,n=20,mult=2.0):
    #https://www.investopedia.com/terms/b/bollingerbands.asp
    #same as Trading View
    basis = SMA(source,n)
    dev = mult * source.rolling(window=n).std(ddof=0)
    upper = basis + dev
    lower = basis - dev
    percentage = (source - lower)/(upper - lower)
    width = (upper - lower)/basis
    return (upper,lower,percentage,width)
    
def EaseOfMovement(high,low,volume,n=14,divisor = 10000):
    #https://www.investopedia.com/terms/e/easeofmovement.asp
    #same as Trading View
    dm = ((high + low)/2).diff()
    br = (volume / divisor) / ((high - low))
    EVM = dm / br 
    EVM_MA = EVM.rolling(window = n).mean()
    return EVM_MA 
    
def CommodityChannelIndex(high,low,close,n = 14): 
    #https://www.investopedia.com/terms/c/commoditychannelindex.asp
    #different from trading view 
    #This indicator is VERY similar to BB%
#    TP = (high + low + close) / 3 
    TP = close
    ma = SMA(TP,n)
    CCI = (TP-ma)/(0.015*(TP-ma).shift(1).abs().rolling(window = n).mean())
    return CCI

def triple_EMA(x,n):
    return EMA(EMA(EMA(x,n),n),n)

def Trix(source,n=18):
    #https://www.investopedia.com/terms/t/trix.asp
    #same as Trading View
    triple = triple_EMA(np.log(source),n)
    trix = 10000*triple.diff()
    return trix

def MoneyFlowIndex(close,high,low,volume,n = 14):
    #https://www.investopedia.com/terms/m/mfi.asp
    #close but still different from Trading View
    TP = (low + high + close)/3
    delta = TP.diff() 
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = (up*volume).ewm(com = n-1).mean()
    roll_down = (down*volume).abs().ewm(com = n-1).mean()
    
    # Calculate the MMFI
    RS = roll_up / roll_down
    MFI = 100.0 - (100.0 / (1.0 + RS))
    return MFI
    

def ChaikinMF(close,high,low,volume,n=20):
    #https://www.tradingview.com/wiki/Chaikin_Money_Flow_(CMF)
    #same as Trading View
    MFVolume = ((2*close-low-high)/(high-low))*volume
    MFVolume = MFVolume.fillna(0)
    CMF = MFVolume.rolling(window = n).sum()/volume.rolling(window = n).sum()
    return CMF

def Ulcer(close,n=14): 
    #https://www.investopedia.com/terms/u/ulcerindex.asp
    #same as Trading View
    max_close = close.rolling(window=n).max()
    pctDrawDown = ((close - max_close)/max_close) * 100
    pctDrawDown2 = pctDrawDown**2
    UI = np.sqrt(((pctDrawDown2.rolling(window = n).mean())))
    return UI
    
def Aroon(high,low,n=25):
    #https://www.investopedia.com/terms/a/aroon.asp
    #same as Trading View
    upper = 100*(high.rolling(window = n+1).apply(np.argmax)+n)/n-100
    lower = 100*(low.rolling(window = n+1).apply(np.argmin)+n)/n-100
    return upper,lower,upper-lower

def AVGTrueRange(close,high,low,n=14):
    #https://www.investopedia.com/terms/a/atr.asp
    #same as Trading View
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1,tr2,tr3],axis = 1)
    ATR = tr.max(axis=1).ewm(com = n-1).mean()
    return ATR

def ChandelierExit(ATR,high,low,n=22,mult=3):
    #https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chandelier_exit
    #same as Trading View (for long, short not tested)
    chandelier_long  = high.rolling(window = n).max() - ATR*mult
    chandelier_short = low.rolling(window = n).min() + ATR*mult
    return chandelier_long,chandelier_short

def VolumeWAveragePrice(close,high,low,volume,n=96):#96 is the number of 15min candles in a day
    #https://www.investopedia.com/terms/v/vwap.asp
    #variation of the VWAP that doesn't reset at midnight
    hlc3 = (high + close + low)/3
    cum_vol = volume.rolling(window = n).sum()
    weighted_price = (hlc3*volume).rolling(window = n).sum()
    return weighted_price/cum_vol

def BearishFractal(h):
    #https://www.investopedia.com/terms/f/fractal.asp
    high = h.copy()
    up_fractal = ((high.shift(-2)  < high) & (high.shift(-1)  < high) & (high.shift(1) < high) & (high.shift(2) < high)) | ((high.shift(-3)  < high) & (high.shift(-2)  < high) & (high.shift(-1) == high) & (high.shift(1) < high) & (high.shift(2) < high))
    up_fractal.iloc[0] = np.nan
    up_fractal.iloc[1] = np.nan
    up_fractal.iloc[-3] = np.nan
    up_fractal.iloc[-2] = np.nan
    up_fractal.iloc[-1] = np.nan
    return up_fractal

def BullishFractal(l):
    #https://www.investopedia.com/terms/f/fractal.asp
    low = l.copy()
    down_fractal = ((low.shift(-2)  > low) & (low.shift(-1)  > low) & (low.shift(1) > low) & (low.shift(2) > low)) | ((low.shift(-3)  > low) & (low.shift(-2)  > low) & (low.shift(-1) == low) & (low.shift(1) > low) & (low.shift(2) > low))
    down_fractal.iloc[0] = np.nan
    down_fractal.iloc[1] = np.nan
    down_fractal.iloc[-3] = np.nan
    down_fractal.iloc[-2] = np.nan
    down_fractal.iloc[-1] = np.nan
    return down_fractal

def pre_processed(d):
    #gathering all indicators into the main dataframe
    
    #filling missing candlesticks with a forward fill
    data = d.copy().drop_duplicates()
    first_timestamp = data.index[0]
    last_timestamp = data.index[-1]
    n_points = (last_timestamp-first_timestamp)//900 #15 min points
    new_index = [first_timestamp + i*900 for i in range(n_points+1)]
    data = data.reindex(index=new_index,method='ffill') #we use the future values if there is a gap
    
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    data['RSI'] = RSI(close,14)
    data['MFI'] = MoneyFlowIndex(close,high,low,volume,14)
    data['ATR'] = AVGTrueRange(close,high,low,14)
    data['MACD'] = MACD(close,24,52,9)
    _,_,data['BB_perc'],data['BB_width'] = BollingerBands(close,20,2)
    data['EOM'] = EaseOfMovement(high,low,volume,14) 
    data['Trix'] = Trix(close,18)
    data['CMF'] = ChaikinMF(close,high,low,volume,20)/2+0.5
    data['Ulcer'] = Ulcer(close,14)
    
    data['EMA24'] = EMA(close,24) # 6 hours for the 15 min chart
    data['EMA48'] = EMA(close,48) # 12 hours for the 15 min chart
    data['EMA_day'] = EMA(close,96) # 1 day for the 15 min chart
    data['EMA_week'] = EMA(close,96*7) # 1 week for the 15 min chart
    data['EMA_month'] = EMA(close,96*30) # 1 month for the 15 min chart
    
    data = data.replace([np.inf, -np.inf], np.nan)
    return data
