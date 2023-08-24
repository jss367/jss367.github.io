pandas to polars


There are no indexes in polars, so if you're relying on one, you'll need to make it a column called "index" (or, more likely, change its name to something else).



No more `.loc` or `.iloc`. You use `.filter`.


Renaming a column is easy -> `.alias`

## Some simple substitutions 
* `.copy` is now `.clone`
* `.fillna` is now `fill_none`


## Strict Typing

You specify return types

    
    building_segmentation_df = building_segmentation_df.with_column(
        pl.col("polygon").apply(
            lambda polygon: convert_pixels_to_sqm(polygon.area, image_gsd=image_gsd),
            return_dtype=pl.Float
        ).alias("area")


You can use the collect method to perform the computations on a LazyFrame and get a DataFrame.
```
df = lf.collect()
```

## Indexing

You can index like so:

```
my_series[0]
```


## Conversions

#### From a pandas DataFrame

Use polars' `from_pandas` to convert a pandas DataFrame to polars format

```python
polars_series = pl.from_pandas(pandas_series)
```

or 

`df = pl.from_pandas(df)`


#### To a Dictionary

It's easy to convert a polars DataFrame to a dictionary, but not a polars series. It's not clear exactly how you want the keys to be. If you want them to be strings, you could do this:

```python
import polars as pl

def pl_series_to_dict(series: pl.Series) -> dict:
    """
    Convert a Polars Series to a dictionary.
    """
    
    # Get the values of the Series
    values = series.to_list()

    # Convert the Series to a dictionary where each index is a key
    result_dict = {str(i): value for i, value in enumerate(values)}

    return result_dict
```
If you want ints, you could do this:
```python
def pl_series_to_dict(series: pl.Series) -> dict:
    """
    Convert a Polars Series to a dictionary.
    """
    
    # Get the values of the Series
    values = series.to_list()

    # Convert the Series to a dictionary where each index is a key
    result_dict = dict(enumerate(values))

    return result_dict
```



#### Data types
```
.cast(int)
```
This will cast the column to the default integer type that the library supports. This might be platform-specific (i.e., it could be 32-bit on some systems and 64-bit on others). If you want to specify specifically which type, you might want to do this:

`.cast(pl.Int64)`

## apply

`apply` operations become operations with `with_columns` and `select`.


## Adding a column to a DataFrame

You can't add a new column to a DataFrame in polars the same way you can in pandas.

In pandas:
```python
df = pd.DataFrame()
df['new_col'] = <whatever you want>
```

In polars:
```python
df = pl.DataFrame()
df = df.with_column(other_df.select("col").with_name("new_col"))
```

Example:
```python
df["clipped_value"] = np.clip(df["value"], np.log(0.01), None)
df_pl = df_pl.with_columns(pl.Series(np.clip(df_pl["value"], np.log(0.01), None)).alias("clipped_value"))
```

## Rename a column within a DataFrame




## General Changes





Using polars vectorized operations like ** for element-wise power


`if row[thing]` becomes `if row[malady].is_not_null().sum() > 0`


This:
```python
    if return_multipliers:
        return pd.Series([multipliers, total_multiplier], index=["multipliers", "rcs"])
    else:
        return pd.Series(total_multiplier)
```

Becomes that:
```python
    if return_multipliers:
        return pl.DataFrame([multipliers, total_multiplier], columns=["multipliers", "rcs"])
    else:
        return pl.DataFrame(total_multiplier, columns=["rcs"])
```





