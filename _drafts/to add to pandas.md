
## Convert pandas Series of lists to pandas Series of tuples.
```
df['items'] = df['items'].apply(lambda x: tuple(x))
```



# General case of setting with copy error:

df[df["col_a"] == True]['col_b'] = True

General solution:

df.loc[df["col_a"] == True, 'col_b'] = True
