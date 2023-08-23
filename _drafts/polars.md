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

## Conversions

Use polars' `from_pandas` to convert a pandas DataFrame to polars format

```python
polars_series = pl.from_pandas(pandas_series)
```

or 

`df = pl.from_pandas(df)`


## apply

`apply` operations become operations with `with_columns` and `select`.


## Adding a column to a DataFrame

You can't add a new column to a DataFrame in polars the same way you can in pandas.

In pandas:
```
df = pd.DataFrame()
df['new_col'] = <whatever you want>
```

In polars:
```
df = pl.DataFrame()
df = df.with_column(other_df.select("col").with_name("new_col"))
```

Example:
```
hail_df["log_staining"] = np.clip(np.log(building_normalized_df["staining"]), np.log(0.01), None)
hail_df_pl.with_columns(pl.Series(np.clip(np.log(building_normalized_df["staining"]), np.log(0.01), None)).alias("log_staining"))
```

## Rename a column within a dataframe

## General Changes





Using polars vectorized operations like ** for element-wise power


`if row[thing]` becomes `if row[malady].is_not_null().sum() > 0`


This:
```
    if return_multipliers:
        return pd.Series([multipliers, total_multiplier], index=["multipliers", "rcs"])
    else:
        return pd.Series(total_multiplier)
```

Becomes that:
```
    if return_multipliers:
        return pl.DataFrame([multipliers, total_multiplier], columns=["multipliers", "rcs"])
    else:
        return pl.DataFrame(total_multiplier, columns=["rcs"])
```





