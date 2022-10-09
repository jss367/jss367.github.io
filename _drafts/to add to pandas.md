
## Convert pandas Series of lists to pandas Series of tuples.
```
df['items'] = df['items'].apply(lambda x: tuple(x))
```
