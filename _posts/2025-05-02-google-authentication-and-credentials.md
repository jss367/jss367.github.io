---
layout: post
title: "Google Authentication and Credentials"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/swallow.jpg"
tags: [Software]
---

Google Cloud authentication can be confusing. This post explains how Application Default Credentials (ADC) work and how to fix common authentication errors.

When you're authenticating with Google Cloud, it's important to be clear on whether you're trying to authenticate as a user or through a service account. Here's some guidance on when to use each:
* **User credentials**: Best for local development and testing
* **Service accounts**: Best for production, CI/CD pipelines, and any automated processes
* **Metadata server**: Best when running on Google Cloud infrastructure

User credentials are tied to an individual identity. Service account credentials represent an automated identity used in production.

## Checking your current credentials

The `GOOGLE_APPLICATION_CREDENTIALS` environment variable tells ADC which credentials file to use. Check if it's set:

```bash
echo $GOOGLE_APPLICATION_CREDENTIALS
```

You might see:
* Nothing (empty) - ADC will look for credentials in the default locations
* Path to service account - e.g., `/path/to/my-service-account.json`
* Path to user credentials - e.g., `/Users/you/.config/gcloud/application_default_credentials.json`

## Where it looks

When you instantiate a client, the library looks for credentials in this order:

- Wherever **`GOOGLE_APPLICATION_CREDENTIALS` env var** is pointing to (user account or service account JSON key)
    
- The **default location**, which is `~/.config/gcloud/application_default_credentials.json`. This is where `gcloud auth application-default login` will create credentials.
    
- **Built-in service account** on GCE/Cloud Run/Cloud Functions via the metadata server. This automatically provides credentials when running on Google Cloud infrastructure—no configuration needed.

## Getting user credentials

```bash
gcloud auth application-default login
```

## Credentials files

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

## Which one am I using?

To check your credentials, you can enter:

```bash
cat $GOOGLE_APPLICATION_CREDENTIALS | grep '"type"'
```

Here are the possible responses:
* User credentials -  `"type": "authorized_user",`
* Service account -  `"type": "service_account",`
* External Account (Workload Identity) -   `"type": "external_account",`
* External Account Authorized User -   `"type": "external_account_authorized_user",`

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

