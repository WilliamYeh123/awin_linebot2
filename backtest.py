import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.dates as mpl_dates
from ta import add_all_ta_features
from sqlalchemy import create_engine
import os, tqdm
engine = create_engine('sqlite:///flaskr.db?check_same_thread=False')
import pandas as pd

# 技術指標 https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html
def add_indicators(df):
	return df.groupby('Tic').apply(lambda tdf: add_all_ta_features(tdf,open='Open',high='High',low='Low',close='Close',volume='Volume'))

# 依據起始日期或最後幾筆，擷取ohlcv
def select(ticker, START_DATE='2020-01-01', LAST=0, con=engine):
	sql = f"SELECT Date,'{ticker}' AS Tic,Open,High,Low,Close,Volume FROM TW{ticker} WHERE Close>0 AND Date>='{START_DATE}';"
	if LAST: sql = f"SELECT DISTINCT(Date),'{ticker}' AS Tic,Open,High,Low,Close,Volume FROM TW{ticker} WHERE Close>0 ORDER BY Date DESC LIMIT {LAST};"
	tdf = pd.read_sql(sql, con).fillna(method='ffill').astype({'Date':'datetime64'}).sort_values('Date')
	# 過濾突波
	tdf[['Open','High','Low','Close']] = tdf[['Open','High','Low','Close']] \
		.apply(lambda s:s[0]*(s.pct_change().clip(-0.2,0.2)+1).fillna(1).cumprod())
	return tdf

with engine.connect() as con:
	TABLES = pd.DataFrame({'name':[TABLE for TABLE in engine.table_names() if TABLE[:2]=='TW']})
	TABLES['min_date'] = TABLES['name'].map(lambda x: con.execute(f'SELECT MIN(Date) FROM {x};').one()[0])

# 抽樣資料
def sample(size=10, START_DATE='2020-01-01', LAST=0) -> pd.DataFrame:
	# 從資料表中篩選出資料足夠者、抽樣並合併表格
	tables = TABLES[TABLES['min_date'] < START_DATE].dropna()['name']
	df = pd.DataFrame()
	for table in tqdm(tables.sample(size)):
		tdf = select(table[2:], START_DATE, LAST)
		df = df.append(tdf)
	# 展開表格，過濾缺漏者，再融解回去
	df = df.drop_duplicates(['Date','Tic'])
	tdf = df.pivot(index='Date',columns='Tic',values='Close').dropna(axis=0)
	tdf = pd.melt(tdf.reset_index(),id_vars='Date')[['Date','Tic']]
	df = df.merge(tdf,on=['Date','Tic'])
	return df

# 蠟燭線
def candle(DOHLC):
	if isinstance(DOHLC, pd.DataFrame): 
		DOHLC = DOHLC[['Date','Open','High','Low','Close']].values
	fig, ax = plt.subplots()
	for D,O,H,L,C in DOHLC:
		D = mpl_dates.date2num(pd.to_datetime(D))
		if C >= O:
			color = 'green'
			lower = O
			height = C - O
		else:
			color = 'red'
			lower = C
			height = O - C
		vline = Line2D(xdata=(D, D), ydata=(L, H), color=color, linewidth=0.5, antialiased=True)
		rect = Rectangle((D - 0.3, lower), width=0.6, height=height, facecolor=color, edgecolor=color)
		ax.add_line(vline)
		ax.add_patch(rect)
	ax.autoscale_view()
	ax.xaxis.set_major_formatter(mpl_dates.DateFormatter('%Y-%m-%d'))
	fig.autofmt_xdate()
	return fig, ax

# 儲存圖片  
def savefig(path, legend=True):
	dirname = os.path.dirname(path)
	if not os.path.exists(dirname):
		os.makedirs(dirname, exist_ok=True)
	if legend: plt.legend()
	plt.tight_layout()
	plt.savefig(path)
	plt.close()
	print(path)

# 窮舉所有組合 [[1,2],[3,4],[5,6]] -> [[1, 3, 5], [1, 3, 6], [1, 4, 5], [1, 4, 6], [2, 3, 5], [2, 3, 6], [2, 4, 5], [2, 4, 6]]
def list_all(lst=[[1,2],[3,4],[5,6]], i=0, tmp=[]):
	j = len(lst) 
	if i>=j: return tmp
	arr = [y for x in lst[i] for y in list_all(lst,i+1,tmp+[x])]
	return [arr[i*j:(i+1)*j] for i in range(len(arr)//j)] if i==0 else arr

# 向上穿越
def upcross(series:pd.Series):
	return (series.shift(1)<0)&(series>0)

# 條件轉持有狀況
def position(long_condition:pd.Series, short_condition:pd.Series):
	series = pd.Series(index=long_condition.index)
	series[long_condition] = True
	series[short_condition]  = False
	return series.fillna(method='ffill').fillna(False)

# 輸入pct_change().fillna(0)轉換成報酬率指標
def calc_stats(rets, risk_free=0):
	cumprod = (rets+1).cumprod()
	cummax = cumprod.cummax()
	stats =  {
		'APV': cumprod.tail(1).item(),
		'SR': (rets.mean()-risk_free)/(rets.std() or 1)*252**0.5,
		'MDD': ((cumprod-cummax)/cummax).min(),
	}
	for k,v in stats.items(): stats[k] = round(v,2)
	return stats

# 策略屬性
class Strategy:
	name = '我的策略'
	ticker = 2330
	position = 'SHORT'
	df = pd.DataFrame()
	trades = pd.DataFrame(columns=['Date','Price','Asset','Action'])
	stats = calc_stats(pd.Series([0]))
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
		self.setup(self.df)
	def setup(self, df=pd.DataFrame()):
		pass
	def buy(self, t, i):
		return False
	def sell(self, t, i):
		return False
	def predict(self, t, i):
		return 0
	def next(self, t, i):
		pass
	def plot(self):
		pass

# 回測框架
class Backtest:
	commision =  0.001425
	risk_free = 0
	result = pd.DataFrame()
	def __init__(self, df=pd.DataFrame(), **kwargs):
		self.df = df
		self.__dict__.update(kwargs)
		self.result = pd.DataFrame(index=self.df['Tic'].unique()) # 交易紀錄矩陣
		# rf = self.df.pivot(index='Date', columns='Tic', values='Close')
		# self.risk_free = (rf.iloc[-1,:]/rf.iloc[0,:]).mean()
	# 檢視不同參數的績效
	def tune(self, _Strategy:Strategy, constraint=lambda row:True, **kwargs) -> pd.DataFrame:
		# 組合轉成列表
		stat_columns = list(Strategy.stats.keys())
		grid = pd.DataFrame(list_all(list(kwargs.values())), columns=kwargs.keys())
		grid[stat_columns] = np.NaN
		for i, row in tqdm(grid.iterrows()):
			if not constraint(row): continue
			tic_stats = []
			for tic, tdf in self.df.groupby('Tic'):
				strategy = _Strategy(df=tdf, **row)
				tic_stats.append(self.backtest(tdf, strategy).stats)
			grid.loc[i,stat_columns] = pd.DataFrame(tic_stats).mean()
		return grid.dropna()
	# 對所有代號進行回測
	def run(self, _Strategy_:Strategy, **kwargs):
		name = kwargs.get('name') or _Strategy_.__name__
		self.result[name] = None
		for tic, tdf in self.df.groupby('Tic'):
			strategy = _Strategy_(df=tdf, ticker=tic, **kwargs)
			strategy = self.backtest(tdf, strategy)
			self.result.loc[tic, name] = strategy            
		return self
	# 單一代號回測
	def backtest(self, df, strategy:Strategy) -> Strategy:
		asset, last_close, trades = 1, np.NaN, []
		position = 'SHORT'
		for i,t in enumerate(df.itertuples()):
			date, close, action = getattr(t, 'Date'), getattr(t, 'Close'), 'HOLD'
			_, buy, sell, pred = strategy.next(t,i), strategy.buy(t,i), strategy.sell(t,i), strategy.predict(t,i)
			if buy and sell and strategy.position=='LONG':
				asset *= close/last_close
			elif strategy.position=='SHORT' and buy:
				asset *= 1-self.commision
				position = strategy.position = 'LONG'
				action = 'BUY'
			elif strategy.position=='LONG' and sell:
				asset *= 1-self.commision
				position = strategy.position = 'SHORT'
				action = 'SELL'
			elif strategy.position=='LONG':
				asset *= close/last_close
			last_close = close
			trades.append([date,close,asset,action,position,pred])
		# 計算績效
		strategy.trades = pd.DataFrame(trades,columns=['Date','Close','Asset','Action','Position','Pred']).set_index('Date')
		strategy.stats = calc_stats(strategy.trades['Asset'].pct_change().fillna(0), self.risk_free)
		return strategy
	# 績效
	def matrix(self, stat='sharpe') -> pd.DataFrame:
		return self.result.apply(lambda s:s.map(lambda x: x.stats[stat]))
	# 預測
	def predict(self, _Strategy_:Strategy) -> pd.DataFrame:
		rows = []
		for tic, result in self.result[_Strategy_.__name__].iteritems():
			row = result.trades.tail(1)
			row.insert(0,'Tic',tic)
			rows.append(row)
		return pd.concat(rows)
	# 特定檔次買賣點比較
	def compare(self, ticker):
		series = self.result.loc[ticker]
		for name in series.index:
			strategy = series[name]
			tdf = strategy.trades
			adf = tdf[tdf['Action']=='BUY']
			plt.plot(adf.index,adf.Asset,'go')
			adf = tdf[tdf['Action']=='SELL']
			plt.plot(adf.index,adf.Asset,'ro')
			tdf['Asset'].plot(label=name)
		plt.legend(loc='upper left')
		plt.xlabel('Asset')
		strategy.trades['Close'].plot(title=ticker, figsize=(16,8), secondary_y=True)
		plt.legend(loc='upper right')

# ============= 範例指標 =============
class KD(Strategy):
	def buy(self, t, i):
		K = t.momentum_stoch_signal
		D = t.momentum_stoch
		return K > D or D < 10
	def sell(self, t, i):
		K = t.momentum_stoch_signal
		D = t.momentum_stoch
		return K < D or D > 90

class RSI(Strategy):
	def buy(self, t, i):
		return t.momentum_rsi > 30
	def sell(self, t, i):
		return t.momentum_rsi < 70

class BB(Strategy):
	def buy(self, t, i):
		return t.Close > t.volatility_bbl
	def sell(self, t, i):
		return t.Close < t.volatility_bbh

class CCI(Strategy):
	def buy(self, t, i):
		return t.trend_cci > 0
	def sell(self, t, i):
		return t.trend_cci < 0

class ADX(Strategy):
	# trend_adx trend_adx_pos trend_adx_neg
	def buy(self, t, i):
		return t.trend_adx_pos > t.trend_adx
	def sell(self, t, i):
		return t.trend_adx_pos < t.trend_adx

class MACD(Strategy):
	def buy(self, t, i):
		return t.trend_macd_signal > t.trend_macd and t.trend_macd < 0
	def sell(self, t, i):
		return t.trend_macd_signal < t.trend_macd and t.trend_macd > 0

class WR(Strategy):
	def buy(self, t, i):
		return t.momentum_wr > -20
	def sell(self, t, i):
		return t.momentum_wr < 80

class SR(Strategy):
	def buy(self, t, i):
		return t.momentum_stoch_signal > t.momentum_stoch and t.momentum_stoch < 20
	def sell(self, t, i):
		return t.momentum_stoch_signal < t.momentum_stoch and t.momentum_stoch > 80

class SMA_CROSS(Strategy):
	def buy(self, t, i):
		return t.trend_sma_fast > t.trend_sma_slow
	def sell(self, t, i):
		return t.trend_sma_fast < t.trend_sma_slow

# 可調參數策略：指數平滑
class EMA_CROSS(Strategy):
	fast = 12
	slow = 26
	def setup(self, df:pd.DataFrame):
		df['fast'] = df['Close'].ewm(span=self.fast).mean()
		df['slow'] = df['Close'].ewm(span=self.slow).mean()
	def buy(self, t, i):
		return t.fast > t.slow
	def sell(self, t, i):
		return t.fast < t.slow

# 動態更新策略：K線撐壓
class KLINE(Strategy):
	def setup(self, df):
		self.last = None
		self.low_list = []
		self.high_list = []
		self.lower_bound = []
		self.upper_bound = []
	def buy(self, t, i):
		return t.Close > self.upper_bound[-1]
	def sell(self, t, i):
		return t.Close < self.lower_bound[-1]
	def next(self, t, i):
		L = self.last
		if L is not None:
			# 前日高點
			if L.High < t.Close:
				self.high_list.append(L.High)
			# 前日低點
			if L.Low < t.Close:
				self.low_list.append(L.Low)
			# 當價格回檔時，吃掉low_list
			if L.Close < t.Close:
				self.low_list = self.low_list[1:]
			# 當價格下跌時，吃掉high_list
			if L.Close > t.Close:
				self.high_list = self.high_list[1:]
		# 若清單被清空，則以nan取代，方便做統計
		self.lower_bound.append(np.mean(self.low_list) if len(self.low_list) else np.nan)
		self.upper_bound.append(np.mean(self.high_list) if len(self.high_list) else np.nan)
		self.last = t
	def plot(self):
		self.trades['H'] = self.upper_bound
		self.trades['L'] = self.lower_bound
		self.trades[['Price', 'H', 'L']].plot(title=self.ticker)

if __name__ == '__main__':
	# from backtest import *
	df = add_indicators(sample(10))
	bt = Backtest(df)
	# 運行不同策略
	bt.run(KD)
	bt.run(RSI)
	bt.run(BB)
	bt.run(CCI)
	bt.run(ADX)
	bt.run(MACD)
	bt.run(WR)
	bt.run(SR)
	bt.run(SMA_CROSS)
	bt.run(EMA_CROSS)
	bt.run(KLINE)
	# 比較績效、產KDE圖
	print('績效矩陣\n', bt.matrix('APV').round(2))
	print('績效列表\n', bt.matrix('APV').mean().round(2) - bt.risk_free)
	bt.matrix('APV').plot.kde()
	savefig('static/images/KDE策略比較.png')
	#  策略自訂繪圖
	bt.result['KLINE'][0].plot()
	savefig('static/images/KLINE策略.png')
	# 第一檔各策略買賣點比較
	bt.compare(bt.result.index[0])
	savefig('static/images/策略比較.png')
	# 找出最佳參數(grid search)
	grid = bt.tune(EMA_CROSS, fast=range(1,13), slow=range(12,27), constraint=lambda x:x.fast<x.slow)
	print('最佳參數\n', grid[grid['APV']==grid['APV'].max()])