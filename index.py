import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')


file_path = r'D:\3 semister\Exiton\Task 4(Predicting Stock Prices Using Sentiment Analysis)\all-data.csv'



try:
    news_data = pd.read_csv(file_path, encoding='latin1')  
except Exception as e:
    print(f"Error loading the CSV file: {e}")


print(news_data.head())  
print(news_data.info())  


news_data.columns = ['Category', 'News']


def clean_text(text):

    text = re.sub(r"[^a-zA-Z\s]", "", text, re.I | re.A)
    text = text.lower().strip()
    return text

news_data['cleaned_news'] = news_data['News'].apply(clean_text)


sia = SentimentIntensityAnalyzer()

news_data['sentiment_score'] = news_data['cleaned_news'].apply(lambda x: sia.polarity_scores(x)['compound'])


sns.histplot(news_data['sentiment_score'], bins=50, kde=True)
plt.title("Sentiment Score Distribution")
plt.show()


dates = pd.date_range(start='2021-01-01', periods=len(news_data), freq='D')
stock_prices = pd.DataFrame({
    'Date': dates,
    'Stock_Price': np.random.randint(100, 200, len(news_data))  
})


stock_sentiment_data = pd.concat([stock_prices.reset_index(drop=True), news_data['sentiment_score']], axis=1)


X = stock_sentiment_data[['sentiment_score']]  
y = stock_sentiment_data['Stock_Price']       

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse}")


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red')
plt.legend()
plt.title('Actual vs Predicted Stock Prices')
plt.show()
