---
layout: post
title: "Google Authentication and Credentials"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/swallow.jpg"
tags: [Software]
---

Google Cloud authentication can be confusing. This post explains how Application Default Credentials (ADC) work and how to fix common authentication errors.

When you're authenticating with Google Cloud, it's important to be clear on whether you're trying to authenticate as a user or through a service account. User credentials are tied to an individual identity and work well for development. Service account credentials represent an automated identity used in production.

When you instantiate a client, the library looks for credentials in this order:

- **`GOOGLE_APPLICATION_CREDENTIALS` env var** pointing to a service-account JSON key
    
- **User credentials** from `gcloud auth application-default login` (in `~/.config/gcloud/application_default_credentials.json`)
    
- **Built-in service account** on GCE/Cloud Run/Cloud Functions via the metadata server

If you `echo $GOOGLE_APPLICATION_CREDENTIALS` it should be `/path/to/your-service-account.json` if it exists.

Below are two examples of what the credentials file might look like, depending on the type:

```
// User credentials example
{
  "account": "",
  "client_id": "[REDACTED_CLIENT_ID].apps.googleusercontent.com",
  "client_secret": "[REDACTED_SECRET]",
  "refresh_token": "[REDACTED_REFRESH_TOKEN]",
  "type": "authorized_user",
  "universe_domain": "googleapis.com"
}
```
```
// Service account example
{
  "type": "service_account",
  "project_id": "[PROJECT_ID]",
  "private_key_id": "[REDACTED_KEY_ID]",
  "private_key": "[REDACTED_PRIVATE_KEY]",
  "client_email": "[SERVICE_ACCOUNT]@[PROJECT_ID].iam.gserviceaccount.com",
  "client_id": "[REDACTED_CLIENT_ID]",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "[REDACTED_CERT_URL]"
}
```
It will either say `"type": "authorized_user"` if it's user credentials or `"type": "service_account"` 


If you ran `gcloud auth application-default login` and haven’t set `GOOGLE_APPLICATION_CREDENTIALS`, ADC will default to your user credentials.

There is a script you can run to show your credentials here: [https://github.com/jss367/scripts/blob/master/check_google_credentials.py](https://github.com/jss367/scripts/blob/master/check_google_credentials.py)

## Which one do I want to use?


Let's walk through an example. Say you've run into this problem:

```
PermissionDenied: 403 Your application is authenticating by using local Application Default Credentials. The documentai.googleapis.com API requires a quota project, which is not set by default.
```

You probably want to switch to your service account.

### Switching to your service account

You can see all your projects with `gcloud projects list`

You might get something like this: 
```
PROJECT_ID       NAME           PROJECT_NUMBER
companydev       companydev     123456789
```
Note that your id is not your number. It's more likely to be a text string.

So you would do this:

```
gcloud iam service-accounts list \
  --project companydev
```

