


sample accuracy?


90% conf


Margin of error = Critical value * Standard deviation of statistic



we want to be within 5% 90% of the time



Sample mean, x"	σx = σ / sqrt( n )




SEx = s / sqrt( n )




Margin of error = Critical value * Standard error of statistic



Confidence interval = sample statistic + Margin of error



The confidence interval is the value plus or minus the margin or error. The margin of error is 





We want to get an idea for how confident we are that the accuracy is between two values. When we say it's 70%, we know it's not EXACTLY 70%. So how sure are we?




sample size = 100

degrees of  freedom = 99

desired_conf_level = .90

(1-0.90) = 0.10

check two-tailed t-distribution table - (or divide by two and one one-tailed)

1.645 - this is using a value of 0.10



what is sample standard deviation... there is none...


conv_int / 2 = 0.90 / 2 = 0.45

z-chart gets 1.65


p-hat value:
num  "events" over num trials
as in, 70 correct over 100 total
so 0.70

0.70+_(1.65)*sqrt(0.70*0.30/100)

0.70 +- 0.07561249896


0.70+_(1.65)*sqrt(0.70*0.30/200)

0.70 +- 0.05346611076



I followed this guide: https://www.statisticshowto.com/probability-and-statistics/confidence-interval/

I decided on a 90% confidence level and got the value of 1.65 from a z-table. Then based on an accuracy of 0.70 and 100 samples, I calculated the confidence interval like so: 0.70+_(1.65)*sqrt(0.70*0.30/100) = 0.70 +- 0.07561249896.


Then I did it again with 200 samples and got the following: 0.70+_(1.65)*sqrt(0.70*0.30/200) = 0.70 +- 0.05346611076

