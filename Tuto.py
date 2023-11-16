import pandas as pd

df=pd.read_csv('data/ForestFiresUCIIrvine/forestfires.csv')
df['day'] = ((df['day'] == 'sun') | (df['day'] == 'sat')| (df['day'] == 'frid'))
print(df['day'])
df=df.rename(columns={'day': 'is_weekend'})
print(df)

#Some encoding: We'll change the months to 1-12 and the days to 0/1 for weekdays/weekend (friday included)
