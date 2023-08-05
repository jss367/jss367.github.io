---
layout: post
title: "JQL Cheat Sheet"
description: "This post is about the Jira Query Language (JQL) and how to use it."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/pallid_cuckoo.jpg"
tags: [Cheat Sheet]
---

Jira Query Language (JQL) is a powerful tool that allows users to perform advanced searches in Jira. This post contains some of my tips and tricks for working with it.

# Access

To get to the JQL editor from any Jira page, click on the "Filters" tab at the top.

<img width="822" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/4b943dbb-3e33-43f3-bddd-d8b36d417fe3">

# Query Structure

The structure of a JQL query is:

* Field: The aspect of the issue you want to query on (e.g., status, assignee, duedate).
* Operator: This defines the relationship between the field and the operand (e.g., =, !=, <, >, IN).
* Operand: The value that is being compared against the field (e.g., "In Progress", currentuser(), "Bug").

You can also combine queries using the `AND` and `OR` operators.

# Fields, Operators, and Operands

Jira provides a lot of different fields you can query on, as well as different operators and operand types. Here are a few examples:

* Fields: project, issuetype, reporter, priority, comment
* Operators: =, !=, <, >, ~ (approximately), IS, IS NOT, IN, NOT IN
* Operands: Text strings (e.g., "In Progress"), functions (e.g., now()), lists (e.g., (HIGH, MEDIUM)), and more.

# Examples

## What are the most recent tickets?

```jql
created >= -30d order by created DESC
```

## Searching by person

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
