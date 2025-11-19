---
layout: post
title: "How to Use CloudWatch"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/clouds.jpg"
tags: [AWS]
---

This post provides a quick guide on the basics of how to work with [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/getting-started/) and, in particular, CloudWatch Logs Insights. CloudWatch Logs Insights is what lets you run SQL-like queries over your logs.

You can open CloudWatch from the AWS console. The opening screen has lots of stuff on it, but we're going to ignore that and look at the menu on the left. You'll see "Log groups" under "Logs". Click on that.

Select the right log group. If it's a Lambda function, it should look something like `/aws/lambda/my-function`. Click on that.

That should open a page that has a lot going on, but this is not how I prefer to look at logs. On the right, you'll see "View in Logs Insights". Click on that.

<img width="632" height="66" alt="image" src="https://github.com/user-attachments/assets/1ca65100-8f78-4215-8afe-bfee1bf22496" />

Here, you can search using CloudWatch Logs Insights query language (Log Insights QL), which you can find out more about [here](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax.html). It looks like this:

```
fields @timestamp, @message, @logStream, @log
| sort @timestamp desc
| limit 10000
```

You can filter on various fields, but you should be aware that there's ANOTHER filter for the date. On the top right of the screen, you'll see this:

<img width="707" height="54" alt="image" src="https://github.com/user-attachments/assets/35abc9ee-1989-43d1-83d5-51d174da8287" />

If you want to go way back in time, click on "Custom" and select your date. Otherwise, searching just using the filter command for dates won't work.

Let's say within your message you have fields called `msg_text` and `s3_key`. You can search for text within them like so:

```
fields @timestamp, @message, @logStream
| parse @message '"msg_text": "*"' as msg_text 
| parse @message '"s3_key": "*"' as s3_key 
| filter msg_text = "Can you find this in the logs?" and s3_key = "path/to/my/file.ext"
| sort @timestamp desc
| limit 100
```

## Cost

Note that queries in Logs Insights are not free. They are billed per GB scanned. That's why it can be good to narrow time windows and log groups when possible.
