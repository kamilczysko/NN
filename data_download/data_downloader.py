import pandas as pa
import requests as req
import json
import time
import datetime

possible_periods = [300, 900, 1800, 7200, 14400, 86400]

def get_history(coin, start, end, period):

    s = str(get_time_stamp(start))
    e = str(get_time_stamp(end))
    p = str(period)
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=BTC_'+coin+'&start='+s+'&end='+e+'&period='+p
    data = json.loads(req.get(url).text)
    if 'error' not in data:
        save_data(coin+'_'+p, data)

def get_time_stamp(data):
    time_stamp = time.mktime(datetime.datetime.strptime(data, "%d.%m.%Y").timetuple())
    return int(time_stamp)

def save_data(file_name, data):
    file = 'marketData/'+file_name+'.csv'
    stack = prepare_data_frames(data)
    stack.to_csv(file ,sep=';', encoding='utf-8')
    print('saved: '+file)

def prepare_data_frames(data):
    t = []
    h = []
    l = []
    o = []
    c = []
    v = []
    wa = []

    for d in data:
        t.append(d['date'])
        h.append(d['high'])
        l.append(d['low'])
        o.append(d['open'])
        c.append(d['close'])
        v.append(d['volume'])
        wa.append(d['weightedAverage'])

    prepared_data = {
        'high': pa.Series(h, index=t),
        'low': pa.Series(l, index=t),
        'open': pa.Series(o, index=t),
        'close': pa.Series(c, index=t),
        'volume': pa.Series(v, index=t),
        'weightedAverage': pa.Series(wa, index=t)
    }

    data_stack = pa.DataFrame(prepared_data)
    return data_stack
