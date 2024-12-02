digiKam Tutorial


# Installing digiKam

Starts with Apple could not verify...


Go to System Preferences > Privacy & Security
Look for text about "digiKam-<version>.pkg" being blocked
Click "Open Anyway"




# Setting up a MySQL Server for Digikam on a Mac


Install it with homebrew: `brew install mysql`


It should tell you this:
```
We've installed your MySQL database without a root password. To secure it run:
    mysql_secure_installation

MySQL is configured to only allow connections from localhost by default

To connect run:
    mysql -u root

To start mysql now and restart at login:
  brew services start mysql
```


Then you'll start the service

`brew services start mysql`



You'll want to secure it, so run:

`mysql_secure_installation`

You'll need to choose a password. For this tutorial, I'll use `mysqlroot`. I recommend storing your password in a password manager, such as BitWarden.


For security, I set a simple password (it's not connected to the Internet), remove anonymous users (it's a security best practice), disallow root login remotely, remove test database, and then reload the privilege tables.



Now it's set up. Open a terminal and log in to MySQL as the root user or the user you created during the setup:

`mysql -u root -p`


Create a database for DigiKam:

```
CREATE DATABASE digikam_core;
CREATE DATABASE digikam_thumbs;
CREATE DATABASE digikam_face;
CREATE DATABASE digikam_similarity;
```


Create a user for DigiKam and grant it privileges on the new database:

* localhost was not working for me, so this is what I did

```
CREATE USER 'digikam_user'@'127.0.0.1' IDENTIFIED BY 'mysqlroot';
GRANT ALL PRIVILEGES ON digikam_core.* TO 'digikam_user'@'127.0.0.1';
GRANT ALL PRIVILEGES ON digikam_thumbs.* TO 'digikam_user'@'127.0.0.1';
GRANT ALL PRIVILEGES ON digikam_face.* TO 'digikam_user'@'127.0.0.1';
GRANT ALL PRIVILEGES ON digikam_similarity.* TO 'digikam_user'@'127.0.0.1';
FLUSH PRIVILEGES;
```

Check your work:

`SELECT User, Host FROM mysql.user;`

The result should look like this:

```
+------------------+-----------+
| User             | Host      |
+------------------+-----------+
| digikam_user     | localhost |
| mysql.infoschema | localhost |
| mysql.session    | localhost |
| mysql.sys        | localhost |
| root             | localhost |
+------------------+-----------+
5 rows in set (0.00 sec)
```


`SHOW DATABASES;`

mysql> SHOW DATABASES;
+--------------------+
| Database           |
+--------------------+
| digikam_core       |
| digikam_face       |
| digikam_similarity |
| digikam_thumbs     |
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
8 rows in set (0.01 sec)


Make sure `digikam_user` is in there. Don't worry about the other stuff.









Now, set up digiKam:


```
Host Name: localhost
User: digikam_user
Password: mysqlroot
Port: 3306 (Should be the default)
Core Db Name: digikam_core
Thumbs Db Name: digikam_thumbs
Face Db Name: digikam_face
Similarity Db Name: digikam_similarity
```

I had a problem

ALTER USER 'digikam_user'@'localhost' IDENTIFIED WITH mysql_native_password BY 'mysqlroot';
FLUSH PRIVILEGES;




## Troubleshooting

Hopefully, you don't have to come down here, but in case you do, I'll suggest some things.

Sometimes, your database won't connect.

<img width="397" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/15a419aa-3a93-47db-9026-c48235b650ce">


Mine is here:

`/System/Volumes/Data/opt/homebrew/etc/my.cnf`


looks like this:

# Default Homebrew MySQL server config
[mysqld]
# Only allow connections from localhost
bind-address = 127.0.0.1
mysqlx-bind-address = 127.0.0.1





You can also test your credentials like so:
`mysql -u digikam_user -p -h localhost -P 3306 --socket=/tmp/mysql.sock`




Check if MySQL is running:
`ps aux | grep mysqld`








ALTER USER 'digikam_user'@'localhost' IDENTIFIED WITH mysql_native_password BY 'mysqlroot666';
FLUSH PRIVILEGES;
















# Set up Mysql Internal

/Users/julius/Library/Application Support/digikam



Your rc file is stored here:


/Users/julius/Library/Preferences/digikamrc


