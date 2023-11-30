---
layout: post
title: "PostgreSQL Cheat Sheet"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/great_crested_grebe.jpg"
tags: [Databases, Python]
---

This notebook contains my cheat sheet for working with PostgreSQL databases.

<b>Table of Contents</b>
* TOC
{:toc}

## Getting Started Once the Database Is Set Up

You can do this either in a jupyter notebook or in a shell.

In a shell, enter `psql`

Note that the user names are case sensitive. So if you set up an account with your username `Julius` it will be `julius`. So you have to log in as follows:

`psql -d postgres -U julius`

You can also log in to the admin account like so:

`psql -d postgres -U postgres`

#### Creating a Table

Let's create a table with PostGIS geometries in it.

`CREATE TABLE IF NOT EXISTS geom_table (id serial primary key, polygon_geom geometry(Polygon, 4326), geometry_geom geometry(Geometry, 4326));`

## Administering the Database

One way to administer your database is with pdAdmin. You'll probably find that for most things you prefer the command line, but for some things it's nice to have a UI.

![image.png](attachment:image.png)

You can see the users here: `julius` and `postgres`. `postgres` is the admin account. 

## Permissions

Here's how to update permissions.

`GRANT UPDATE, DELETE, REFERENCES, INSERT, TRUNCATE, SELECT, TRIGGER ON TABLE public.geom_table TO julius;`

Again, you can do this in the pgAdmin.

![image.png](attachment:image.png)

You'll also have to give permission to sequences: `GRANT UPDATE, USAGE, SELECT ON SEQUENCE public.geom_table_id_seq TO julius;`




#### Looking Around

`\dt`

![image-2.png](attachment:image-2.png)

Query a table

`select * from geom_table limit 10`

## Entering into database

Now let's add some stuff to our table


```python
insert into geom_table (polygon_geom, geometry_geom) values ('0103000020E61000000100000005000000EF30A5DE4B855EC0C3308F769AB347406ED22F8149855EC0DD60A6D99AB347404AB842264A855EC035C13C23A2B34740CD16B8834C855EC01B9125C0A1B34740EF30A5DE4B855EC0C3308F769AB34740', '0103000020E61000000100000005000000EF30A5DE4B855EC0C3308F769AB347406ED22F8149855EC0DD60A6D99AB347404AB842264A855EC035C13C23A2B34740CD16B8834C855EC01B9125C0A1B34740EF30A5DE4B855EC0C3308F769AB34740')
```




    '0103000020E61000000100000005000000EF30A5DE4B855EC0C3308F769AB347406ED22F8149855EC0DD60A6D99AB347404AB842264A855EC035C13C23A2B34740CD16B8834C855EC01B9125C0A1B34740EF30A5DE4B855EC0C3308F769AB34740'



I don't like doing a lot from the command line tools. It doesn't give you enough feedback if you make a mistake. For example, if you try to insert into a table that doesn't exist (perhaps you had a typo in the name) it doesn't give you an error.

`INSERT INTO geom_table (polygon_geom, geometry_geom) VALUES (GeomFromText('POLYGON ((4 5, 5 5, 5 4, 4 4, 4 5))'), GeomFromText('POLYGON ((4 5, 5 5, 5 4, 4 4, 4 5))'))`


Note that you can also get weird errors. For example, when I run this:
```
INSERT INTO Customers(name, email, entry_num, age) VALUES ("John Smith", "js@gmail.com", 100, 25);
```

The error is weird

![image.png](attachment:image.png)

The solution is to change your single quotes to double quotes.

## Some samples

```sql
CREATE TABLE Customers (
  name text,
  email text,
  entry_num int,
  age int
);

INSERT INTO Customers(name, email, entry_num, age) VALUES ("John Smith", "js@gmail.com", 100, 25);
INSERT INTO Customers(name, email, entry_num, age) VALUES ("Karen Smith", "ks@gmail.com", 101, 42);
INSERT INTO Customers(name, email, entry_num, age) VALUES ("John Smith", "js@gmail.com", 102, 25);



select entry_num, count(*) from customers
group by entry_num
having count(*)>1
```

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

## Using pandas

You can load results into a DataFrame like so:
```python
with db_session(engine=my_engine) as sess:
    records = sess.execute(query_string)
df = pd.DataFrame(records)
```
But you can also do it directly using `read_sql`:
```python
with db_session(engine=my_engine) as sess:
    df = pd.read_sql(query_string, sess.bind)
```

## Useful commands

`update my_table set deleted=now() where ...`

## Reminders

If you're coming from a Python world, remember, single quotes and double quotes are not interchangeable!

## Jsonb

The `->>` operator gets the JSONB array element as a text, while the `->` operator gets it as JSONB. If you want to process the element as JSONB, use `->`; if you need it as text for comparison or other operations, use `->>`.

