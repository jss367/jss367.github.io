

Temporary credentials

Let's say you're using AWS vault... thus, all your credentials are temporary. It works by injects temporary AWS credentials (access key, secret key, and session token) into your environment.




Your long-term credentials might be stored in ~/.aws/config. There's no API keys in there though.



The way to get your credentials inside phoenix is to launch it inside aws vault








