


Let's say you're on a VPN and having DNS issues.


This uses the DNS lookup tool **`dig`** (“Domain Information Groper”) to resolve the hostname `mycompany.abc123.us-west-2.rds.amazonaws.com` into its corresponding **IP address(es)**.

So you find the hostname that's giving a problem. Let's say it's your company's connection to metabase. So the hostname is `metabase.companyinternal.com`

The response will include a bunch of information. The most important part is the ANSWER SECTION

The IP addresses will all be within the private ranges (10.x.x.x). It might look like this:

metabase.companyinternal.com. 60	IN	A	10.0.1.11
metabase.companyinternal.com. 60	IN	A	10.0.2.22
metabase.companyinternal.com. 60	IN	A	10.0.3.33

You can also use `nslookup`, but `dig` provides more information. It's generally been replaced by `dig`.


Your Python uses a different resolver.
