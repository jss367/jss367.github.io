
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


## Updating

with db_session(engine=get_engine("bvds")) as sess:
    records = sess.query(MyTable).filter_by(id = 12345)
rec = records.first()
rec.my_column = 'new_value'
with db_session(engine=get_engine("bvds")) as sess:
    sess.merge(rec)
    try:
        sess.commit()
    except Exception as ex:
        print(ex)
        sess.rollback()

## Useful commands:


`update my_table set deleted=now() where ...`


If you're using an ORM and want to update an object, you'll have to get the record, then update it and run `merge`.

