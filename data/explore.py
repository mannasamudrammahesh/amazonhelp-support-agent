import pandas as pd
import numpy as np

df = pd.read_csv('data/twcs/twcs.csv', nrows=200000, low_memory=False, dtype=str)
print('Columns:', list(df.columns))
print('Shape:', df.shape)

amazon = df[df['author_id'] == 'AmazonHelp']
print(f'\nAmazonHelp outbound rows in first 200k: {len(amazon)}')

amazon_reply_ids = set(amazon['in_response_to_tweet_id'].dropna())
customer_msgs = df[df['tweet_id'].isin(amazon_reply_ids)]
print(f'Customer msgs Amazon replied to (first 200k): {len(customer_msgs)}')

if len(customer_msgs) > 0:
    print('\nSample customer messages:')
    for _, r in customer_msgs.sample(min(10, len(customer_msgs)), random_state=42).iterrows():
        text = str(r.get('text', ''))[:100]
        print(f'  {text}')
