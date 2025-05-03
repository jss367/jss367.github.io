---
layout: post
title: "Google Authentication and Credentials "
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/swallow.jpg"
tags: [Software]
---

Google Cloud authentication can be confusing. This post explains how Application Default Credentials (ADC) work and how to fix common authentication errors.

When you're authenticating with Google Cloud, it's important to be clear on whether you're trying to authenticate as a user or through a service account.

When you instantiate a client, the library looks for credentials in this order:

- **`GOOGLE_APPLICATION_CREDENTIALS` env var** pointing to a service-account JSON key
    
- **User credentials** from `gcloud auth application-default login` (in `~/.config/gcloud/application_default_credentials.json`)
    
- **Built-in service account** on GCE/Cloud Run/Cloud Functions via the metadata server

If you `echo $GOOGLE_APPLICATION_CREDENTIALS` it should be `/path/to/your-service-account.json` if it exists.

That json file that contains your credentials and looks something like this:

{
  "account": "",
  "client_id": "[REDACTED_CLIENT_ID].apps.googleusercontent.com",
  "client_secret": "[REDACTED_SECRET]",
  "refresh_token": "[REDACTED_REFRESH_TOKEN]",
  "type": "authorized_user",
  "universe_domain": "googleapis.com"
}

It will either say "type": "authorized_user" if it's user credentials or `"type": "service_account"` 


If you ran `gcloud auth application-default login` and you haven’t set `GOOGLE_APPLICATION_CREDENTIALS` then it should be your user credentials.

Here's a little script to see your credentials:

```
import google.auth
from google.oauth2 import service_account
from google.oauth2 import credentials as oauth2_creds
from google.auth import compute_engine

# This finds whatever ADC would use
creds, project = google.auth.default()

# Inspect the type
print("Credential type:", type(creds))

# Or more precisely:
if isinstance(creds, service_account.Credentials):
    print("→ Using a service account key file")
elif isinstance(creds, oauth2_creds.Credentials):
    print("→ Using user (gcloud) credentials")
elif isinstance(creds, compute_engine.Credentials):
    print("→ Using Compute Engine / metadata-server credentials")
else:
    print("→ Some other credential type:", creds)
```

## Which one do I want to use?


Let's walk through an example. Say you've run into this problem:

```
PermissionDenied: 403 Your application is authenticating by using local Application Default Credentials. The documentai.googleapis.com API requires a quota project, which is not set by default.
```

You probably want to switch to your service account.

### Switching to your service account

You can see all your projects with `gcloud projects list`

You might get something like this: 

PROJECT_ID       NAME           PROJECT_NUMBER
companydev       companydev     123456789

Note that your id is not your number. It's more likely to be a text string.

So you would do this:

```
gcloud iam service-accounts list \
  --project companydev
```

