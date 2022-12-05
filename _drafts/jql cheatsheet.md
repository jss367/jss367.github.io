
```jql
created >= -30d order by created DESC
```


```jql
assignee = <type their name and a reference number will appear>.  
```


```jql
assignee = <hex_num> AND status not in (done) and created >= -100d
```
  
## Search by board and title
``jql
project = MY_TEAM and summary ~ "my_topic*"
```
