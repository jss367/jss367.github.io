---
layout: post
title: "SQLAlchemy and PostgreSQL Cheat Sheet"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/pallid_cuckoo.jpg"
tags: [Databases, Python]
---

This post is about SQLAlchemy and working with ORMs. It's is hiding here while it's in draft.

All of engines, connections, and sessions have a `.execute` method. The results are all the same for simple select queries, but can vary as you do more complex operations.

<b>Table of Contents</b>
* TOC
{:toc}


```python
from sqlalchemy import create_engine, inspect
from sqlalchemy.sql import text
```

# Engines

An engine is how SQLAlchemy communicates with a database.

## Creating an Engine

It's on the form: `engine = create_engine('postgresql://<user>:<password>@<hostname>:<port (optional)>/<database_name>')`


It might look like this:


```python
engine = create_engine('postgresql://julius:post123@localhost:5432/postgres')
```

## Exploring the Database with an Engine

You can use `inspect` to learn more about your database.


```python
inspector = inspect(engine)
```


```python
inspector.get_table_names()
```




    ['customers', 'spatial_ref_sys', 'geom_table']




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



## Raw Queries with an Engine

Now let's start querying


```python
query_string = 'select * from customers'
```


```python
response = engine.execute(query_string)
```

You get a `LegacyCursorResult` object as a response.


```python
response
```




    <sqlalchemy.engine.cursor.LegacyCursorResult at 0x20065e95160>




```python
result = response.fetchall()
```


```python
result
```




    [('John Smith', 'js@gmail.com', 100, 25),
     ('Karen Smith', 'ks@gmail.com', 101, 42),
     ('John Smith', 'js@gmail.com', 102, 25)]



You can also use a `TextClause`


```python
statement = text("""select * from customers""")
```


```python
statement
```




    <sqlalchemy.sql.elements.TextClause object at 0x0000020064993D00>



You can see what the query is by printing the statement.


```python
print(statement)
```

    select * from customers
    


```python
response = engine.execute(statement)
```


```python
result = response.fetchall()
```


```python
result
```




    [('John Smith', 'js@gmail.com', 100, 25),
     ('Karen Smith', 'ks@gmail.com', 101, 42),
     ('John Smith', 'js@gmail.com', 102, 25)]



# ORMs

You can also use an ORM.


```python
from geoalchemy2 import Geometry
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import INTEGER
from sqlalchemy.orm import declarative_base, sessionmaker
```


```python
Base = declarative_base()

class GeomTable(Base):
    __tablename__ = "geom_table"

    id = Column("id", INTEGER, primary_key=True)
    polygon_geom = Column("polygon_geom", Geometry("POLYGON", srid=4326))
    geometry_geom = Column("geometry_geom", Geometry("GEMOETRY", srid=4326))

```

When you're working with ORMs, you generally want to work with Sessions instead of engines.

Create your engine as before.


```python
engine = create_engine('postgresql://julius:post123@localhost:5432/postgres')
```

Then create a Session.


```python
Session = sessionmaker(engine)
```


```python
session = Session()
```

### Retrieving a Record

ORMs can be used to retrieve records.


```python
response = session.query(GeomTable).filter_by(id=2)
```


```python
response
```




    <sqlalchemy.orm.query.Query at 0x20068f395b0>



To get the record, you could do `.first()`, `.get(2)`, .`all()`, or `.one()`

`.all()` returns a list


```python
response.all()
```




    [<__main__.GeomTable at 0x20068f75ca0>]



`.distinct()` returns another query, so you would need to call `.all()` on it.


```python
response.distinct()
```




    <sqlalchemy.orm.query.Query at 0x20068f75a30>



`.first()` returns a single value (the first one), not in a list


```python
response.first()
```




    <__main__.GeomTable at 0x20068f75ca0>




```python
result = response.first()
```


```python
result
```




    <__main__.GeomTable at 0x20068f75ca0>




```python
result.__dict__
```




    {'_sa_instance_state': <sqlalchemy.orm.state.InstanceState at 0x20068f75c70>,
     'geometry_geom': <WKBElement at 0x20068f75be0; 0103000020e61000000100000005000000ef30a5de4b855ec0c3308f769ab347406ed22f8149855ec0dd60a6d99ab347404ab842264a855ec035c13c23a2b34740cd16b8834c855ec01b9125c0a1b34740ef30a5de4b855ec0c3308f769ab34740>,
     'polygon_geom': <WKBElement at 0x20068f75c40; 0103000020e61000000100000005000000ef30a5de4b855ec0c3308f769ab347406ed22f8149855ec0dd60a6d99ab347404ab842264a855ec035c13c23a2b34740cd16b8834c855ec01b9125c0a1b34740ef30a5de4b855ec0c3308f769ab34740>,
     'id': 2}



There's also a [good SO post on using `filter` or `filter_by`](https://stackoverflow.com/questions/2128505/difference-between-filter-and-filter-by-in-sqlalchemy).


#### Sessions and raw text

Just because you're using a session doesn't mean you can't use raw text. It's quite easy.


```python
query_string = text('select * from geom_table where id = 3')
```


```python
with Session() as session:
    record = session.query(query_string)
```


```python
response = engine.execute(query_string)
```


```python
result = response.fetchall()
```


```python
result
```




    [(3, '0103000020E610000001000000050000000000000000001040000000000000144000000000000014400000000000001440000000000000144000000000000010400000000000001040000000000000104000000000000010400000000000001440', '0103000020E610000001000000050000000000000000001040000000000000144000000000000014400000000000001440000000000000144000000000000010400000000000001040000000000000104000000000000010400000000000001440')]



Usually you'll want it in wkt format.


```python
query_string = 'select st_astext(polygon_geom) from geom_table where id = 3'
```


```python
response = engine.execute(query_string)
```


```python
result = response.fetchall()
```


```python
result
```




    [('POLYGON((4 5,5 5,5 4,4 4,4 5))',)]



#### `sess.query` vs `sess.execute`

## Creating a Record


```python
gt = GeomTable()
```


```python
gt.polygon_geom = 'POLYGON ((4 5, 5 5, 5 4, 4 4, 4 5))'
gt.geometry_geom = 'POLYGON ((4 5, 5 5, 5 4, 4 4, 4 5))'
```


```python
with Session() as session:
    session.add(gt)
    session.commit()
```

# Connections

You can also use a connection to execute a SQL query. Using a connection allows us to create transactions where either all the commands or, if there's an error, roll it all back.


You can also make a connection:
```
conn = engine.connect()
conn.execute(...)
```



```
trans = conn.begin()
conn.execute('INSERT INTO MY_TABLE ...)
trans.commit()
```

```
with engine.connect() as con:

    response = con.execute(statement)
```


```python
with engine.connect() as con:

    response = con.execute("select * from customers")
```


```python
response
```




    <sqlalchemy.engine.cursor.LegacyCursorResult at 0x20068fa2e50>




```python
result = response.fetchall()
result
```




    [('John Smith', 'js@gmail.com', 100, 25),
     ('Karen Smith', 'ks@gmail.com', 101, 42),
     ('John Smith', 'js@gmail.com', 102, 25)]



# Updating Databases

If you're using an ORM and want to update an object, you'll have to get the record, then update it and run `merge`.

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

# Synactic Sugar and Tricks

#### Get the date from a timestamp
* `date(my_timestamp)`


```python

```
