import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('data/ForestFiresUCIIrvine/forestfires.csv')
df['day'] = ((df['day'] == 'sun') | (df['day'] == 'sat')| (df['day'] == 'frid'))
df=df.rename(columns={'day': 'is_weekend'})
features = df.drop(['area', 'is_weekend', 'month'], axis = 1)
labels = list(df['area'])
#Some encoding: We'll change the months to 1-12 and the days to 0/1 for weekdays/weekend (friday included)
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size = 0.2, random_state = 42)

X_train_features=[]
for i, row in X_train.iterrows():
	print(row)
	fds
print(y_train)

clf=LogisticRegression().fit(X_train, y_train)