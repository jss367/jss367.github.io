


If you're moving data between places on gcloud, here are the steps:

1. Find the IP address of the destination. If you're moving file `my_docs.tar.gz` to instance `my_instance` at location `/path/to/docs` and you are user `jsimonelli` and working in project `my_project` and zone `europe-west4-b`.


```
gcloud compute instances describe my_isntance \
  --zone=europe-west4-b \
  --project=my_project \
  --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
```

You should get back an IP address. Let's assume it's 123.1.1.1

scp my_docs.tar.gz jsimonelli@123.1.1.1:/path/to/docs

Or you could also use the built-in gcloud command:

gcloud compute scp my_docs.tar.gz my_instance:/path/to/docs


You might be a permission denied error. In that case:

Go to other machine and add the source machine public key to known hosts:

On source machine:

`cat ~/.ssh/id_rsa.pub`



