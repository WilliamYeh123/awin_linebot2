from backtest import engine, Backtest, Strategy, select, add_indicators, BB, savefig, pd, candle
import requests
from sqlalchemy import Column, TEXT, orm

tics = ['2303','2330']
data = pd.concat([select(tic,LAST=90) for tic in tics])
data = add_indicators(data)

class Fiveline(Strategy):
    def setup(self,df):
        y = df['Close'].reset_index(drop=True)
        X = pd.Series(range(len(df)))
        mX, my = X.mean(), y.mean()
        beta = (y-my)*(X-mX)/((X-mX)**2).sum()
        alpha = my-mX*beta
        self.mid = X*beta+alpha
        self.std = (y-my).std()
    def buy(self,t,i):
        return t.Close>self.mid[i]-self.std
    def sell(self,t,i):
        return t.Close<self.mid[i]+self.std

class Fiveline_2std(Fiveline):
    def buy(self,t,i):
        return t.Close>self.mid[i]-2*self.std


#123
bt = Backtest(data)
bt.run(Fiveline)
bt.run(Fiveline_2std)
bt.run(BB)
bt.matrix('APV')
bt.compare(bt.result.index[0])
# savefig('train/compare.png')

Base = orm.declarative_base()
class User(Base):
	__tablename__ = 'User'
	account = Column(TEXT, primary_key = True)
	password = Column(TEXT)
	name = Column(TEXT)
	auth = Column(TEXT)
	token = Column(TEXT)
class BidOrder(Base):
	__tablename__ = 'BidOrder'
	token = Column(TEXT, primary_key = True)
	ticker = Column(TEXT, primary_key = True)
	datetime = Column(TEXT, primary_key = True)
Base.metadata.create_all(engine)
db = orm.Session(engine)

NAME = 'William'
PASSWORD = '4795e9e44a96c607fd36f94de89b7d76'
LINE_TOKEN = 'hpEeLWBBW3HCwgirFWfsg7n4GmqQq34pPc/KJJtpQz3wGPBmo0QZkTR/cmke7CkCPubFQIEqBfpZ+sdngdKsok/2VNU6fZIN8DrCwLAkrO03ZUjNpxRB3LQA2Pi4MfPlmLT7eaT36chuOxlYVKhgJwdB04t89/1O/w1cDnyilFU='
if not db.query(User).filter(User.token==LINE_TOKEN).first():
	db.add(User(account=NAME,password=PASSWORD,name=NAME,auth='line',token=LINE_TOKEN))
	db.commit()

def push_linebot():	
	df = bt.predict(Fiveline)
	# 遇到買點進行推播(Position SHORT->LONG)
	should_buy = df.loc[df['Action']=='BUY', 'Tic']
	should_sell = df.loc[df['Action']=='SELL', 'Tic']
	for (token,) in db.query(User.token).filter(User.auth=='line').all():
		msg = '買' + ','.join(should_buy) + ' 賣' + ','.join(should_sell)
		requests.get(f'https://trading-vue-flask.herokuapp.com/user/{token}/{msg}')
	return df