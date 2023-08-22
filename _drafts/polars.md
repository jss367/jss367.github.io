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





