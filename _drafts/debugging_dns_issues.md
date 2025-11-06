


Let's say you're on a VPN and having DNS issues.


This uses the DNS lookup tool **`dig`** (“Domain Information Groper”) to resolve the hostname `mycompany.abc123.us-west-2.rds.amazonaws.com` into its corresponding **IP address(es)**.

So you find the hostname that's giving a problem. Let's say it's your company's connection to metabase. So the hostname is `metabase.companyinternal.com`

The response will include a bunch of information. The most important part is the ANSWER SECTION

The IP addresses will all be within the private ranges (10.x.x.x). It might look like this:

| Hostname | TTL | Class | Type | IP Address |
|----------|-----|-------|------|------------|
| metabase.companyinternal.com. | 60 | IN | A | 10.0.1.11 |
| metabase.companyinternal.com. | 60 | IN | A | 10.0.2.22 |
| metabase.companyinternal.com. | 60 | IN | A | 10.0.3.33 |

You can also use `nslookup`, but `dig` provides more information. It's generally been replaced by `dig`.


Your Python uses a different resolver.

What's going on is a split DNS mismatch.

You have two different resolvers:

nslookup and system resolver.  (what curl/Python use) cannot resolve it at all.


On macOS, nslookup and the system resolver can consult different resolver “scopes.” Your VPN is likely pushing a private DNS server, but the system isn’t using it for companyinternal.com.



