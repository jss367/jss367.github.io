


let's say your ssh or scp command doesn't work - Permission denied (publickey)


scp my_file.tar.gz jsimonelli@123.456.789.111:/home/

try just sshing there

ssh jsimonelli@123.456.789.111:/home/


ssh jsimonelli@123.456.789.111 - Permission denied (publickey)

to get a verbose output you can do 

ssh -v jsimonelli@34.141.181.199 - Permission denied (publickey)


ssh-copy-id jsimonelli@123.456.789.111




cat /etc/ssh/sshd_config

cat /etc/ssh/sshd_config | grep PubkeyAuthentication


gcloud compute scp --recurse my_file.tar.gz --zone europe-west4-c fundy:/home/


gcloud compute ssh --zone europe-west4-c fundy


Next, add the contents of the public key file into ~/.ssh/authorized_keys on the remote site (the file should be mode 600).


cat ~/.ssh/id_rsa.pub | ssh user@hostname 'tee -a .ssh/authorized_keys'



cat ~/.ssh/id_rsa.pub | ssh jsimonelli@123.456.789.111 'tee -a .ssh/authorized_keys'



gcloud compute ssh --zone europe-west4-c fundy




scp my_file.tar.gz jsimonelli@123.456.789.111:/home/


