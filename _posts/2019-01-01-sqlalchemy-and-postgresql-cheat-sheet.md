---
layout: post
title: "SQLAlchemy and PostgreSQL Cheat Sheet"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/pallid_cuckoo.jpg"
tags: [Databases, Python]
---

This notebook is hiding here while it's in draft.


```python
import sqlalchemy
from sqlalchemy.sql import text
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey
from sqlalchemy import inspect
```


```python

```

An engine is how SQLAlchemy communicates with a database.

If you have a query like:

```sql
query_string = f"""
select * from my_table where col = 123
"""
```
```
engine.execute('SELECT * FROM ...')
```

```
engine.table_names()
```

You can also make a connection:
```
conn = engine.connect()
conn.execute(...)
```

This allows us to create transactions where either all the commands or there's an error and it all rolls back.

```
trans = conn.begin()
conn.execute('INSERT INTO MY_TABLE ...)
trans.commit()
```

But you don't actually need to do this.

You could also make a `sessionmaker`.


```python

```



You can write a function that queries the database:

```
record = query_db(query_string, user, password, host, dbname="bvds")
```

Or you could use an ORM

If it's a raw string, you can just do this:

```sql
with db_session(engine=get_engine('bvds', db_name='bvds')) as sess:
    records = (sess.execute(query_string))
```

Note that you have to use execute

With an ORM, you can see if a record exists

## Querying

session.query(My_Table).filter_by(name='My Name')


filter vs filter_by: `https://stackoverflow.com/questions/2128505/difference-between-filter-and-filter-by-in-sqlalchemy`

x = session.query(Customers).get(2)

To get the record, you could do `.first()`, `.get(2)`, or `.one()`


```python

```

#### Creating an engine


```python

engine = create_engine('postgresql://julius:post123@localhost:5432/postgres')
```


```python

```


```python

```

You can use `inspect` to learn more about your database.


```python
inspector = inspect(engine)
```


```python
inspector.get_table_names()
```




    ['customers']




```python

```


```python

```


```python
columns = inspector.get_columns('customers')
```


```python
columns
```




    [{'name': 'name',
      'type': TEXT(),
      'nullable': True,
      'default': None,
      'autoincrement': False,
      'comment': None},
     {'name': 'email',
      'type': TEXT(),
      'nullable': True,
      'default': None,
      'autoincrement': False,
      'comment': None},
     {'name': 'entry_num',
      'type': INTEGER(),
      'nullable': True,
      'default': None,
      'autoincrement': False,
      'comment': None},
     {'name': 'age',
      'type': INTEGER(),
      'nullable': True,
      'default': None,
      'autoincrement': False,
      'comment': None}]




```python
[c['name'] for c in columns]
```




    ['name', 'email', 'entry_num', 'age']



## Querying

#### sqlalchemy.orm.query.Query

can do `.all()`
* returns list of query results

`[(3,)]`

`.distinct()`
* returns another query
* would need to call `.all()` on it

`.first()`
* Return single value, not in a list

`(3,)`

#### `sess.query` vs `sess.execute`


```python

```

#### Use the engine directly

Now let's start querying


```python
response = engine.execute('select * from customers')
```


```python
response
```




    <sqlalchemy.engine.cursor.LegacyCursorResult at 0x23da27cfd30>




```python
result = response.fetchall()
```


```python
result
```




    [('John Smith', 'js@gmail.com', 100, 25),
     ('Karen Smith', 'ks@gmail.com', 101, 42),
     ('John Smith', 'js@gmail.com', 102, 25)]




```python

```


```python

```


```python
query_string = f"""
select * from my_table where col = 123
"""
```


```python

```


```python
statement = text("""select * from customers""")
```


```python
statement
```




    <sqlalchemy.sql.elements.TextClause object at 0x0000023DA265B3D0>




```python
str(statement)
```




    'select * from customers'



You can also make a connection

```
conn = engine.connect()
conn.execute(...)
```

If you use a connection it's safer (I think). This allows us to create transactions where either all the commands or there's an error and it all rolls back.


```python
with engine.connect() as con:

    response = con.execute(statement)
```

```
trans = conn.begin()
conn.execute('INSERT INTO MY_TABLE ...)
trans.commit()
```

But you don't actually need to do this.


```python
result = response.fetchall()
```


```python
result
```




    [('John Smith', 'js@gmail.com', 100, 25),
     ('Karen Smith', 'ks@gmail.com', 101, 42),
     ('John Smith', 'js@gmail.com', 102, 25)]



You could also skip the TextClause  and type it directly


```python
with engine.connect() as con:

    response = con.execute("select * from customers")
```


```python
result = response.fetchall()
result
```




    [('John Smith', 'js@gmail.com', 100, 25),
     ('Karen Smith', 'ks@gmail.com', 101, 42),
     ('John Smith', 'js@gmail.com', 102, 25)]




```python

```


```python

```


```python

```

Note that you can also get weird errors. For example, when I run this:
```
INSERT INTO Customers(name, email, entry_num, age) VALUES ("John Smith", "js@gmail.com", 100, 25);
```

The error is weird

![image.png](attachment:image.png)

The solution is to change your single quotes to double quotes.

#### Sessionmaker

## Some samples

#### Return all the names that are duplicated and the number of entries

```sql
select name, email, count(*)
from Customers
group by name, email
having count(*) > 1
```

Another way

```sql
with subquery as (
select name, email, entry_num, row_number() over(partition by email) as rk
from Customers
)
select name, email, entry_num 
from subquery
where rk = 1
```

#### Select all rows that are part of a duplicate

```sql
select t.* from Customers as t
inner join
( select name, email, age, count(*) from Customers
group by email, name, age
having count(*) >= 2) as tt
on t.email=tt.email and t.name=tt.name and t.age=tt.age
```

#### Get the first entry of every person

```sql
select name, email, age, min(entry_num) as entry_num from Customers group by email, name, age
```

## Using ORMs


```python
with session(engine=engine) as sess:
    records = (sess.execute(query_string))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In [1], line 1
    ----> 1 with session(engine=engine) as sess:
          2     records = (sess.execute(query_string))
    

    NameError: name 'session' is not defined



## Updating Records

```sql
with session(engine=engine) as sess:
    records = sess.query(MyTable).filter_by(id = 12345)
rec = records.first()
rec.my_column = 'new_value'
with session(engine=engine)) as sess:
    sess.merge(rec)
    try:
        sess.commit()
    except Exception as ex:
        print(ex)
        sess.rollback()
```

Or you could pass one sess to the whole thing.

## Useful commands

`update my_table set deleted=now() where ...`

If you're using an ORM and want to update an object, you'll have to get the record, then update it and run `merge`.

## Synactic Sugar and Tricks

#### Get the date from a timestamp
* `date(my_timestamp)`


```python

```

## SQLAlchemy Commands

#### Filter in Python list

session.query(MyTable).filter(MyTable.id.in_(tuple(my_list))).all()

