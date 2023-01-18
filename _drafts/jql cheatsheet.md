
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
```jql
project = MY_TEAM and summary ~ "my_topic*"
```

## What did X do in the last year?

```jql
created >= -365d AND assignee = <X> AND status = Done order by created DESC
```

## Searching by ticket number

```jql
issuekey = 'MY_TICKET_1234'
```
