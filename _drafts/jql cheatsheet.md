
Jira Query Language (JQL) is a powerful tool that allows you to perform advanced searches in Jira based on various parameters. This post contains some of my tips and tricks for working with it.

To get to the JQL editor, click on the "Filters" tab at the top.

```jql
created >= -30d order by created DESC
```


```jql
assignee = <type their name and a reference number will appear>.  
```
## What is X working on?

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

## Works for any current user:
```jql
assignee = currentUser() AND status = Done AND updatedDate > "-90d" order by updated desc
```
## What did this team do this year?

```jql
project = "Machine Learning" and status = Done AND updatedDate >= startOfYear()
```
