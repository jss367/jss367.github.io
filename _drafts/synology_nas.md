synology


Use Synology Drive


## Location


You'll have files in `/Users/julius/Library/CloudStorage/SynologyDrive-Mac`

Note that the Library folder is hidden by default


## Backups

turn off versions, do snapshots instead


## Freeing up space


You also have mapped a network drive, where you can go to /home
/home is syncd to SynologyDrive-Mac, so they should be the same

This doesn't have disk_usage

Right now, it's pretty buggy on Mac, so sometimes I have to turn it off and turn it on again. But when it's working, In Finder you should see SynologyDrive in your Locations section. If you don't, try tioatioa. From there, you can right-click a folder and you will see the synology actions near the bottom.

<img width="277" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/07af35aa-ec5d-4b6a-8da5-230de3654b9d">
<img width="243" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/e63417a2-cd74-4df0-856f-2834e3d282cf">



Sometimes you'll be in your SynologyDrive folder and the cloud options will go away. The best thing to do in that case is to go to the Synology Drive Client and pause it, then start it again. Then go into a new finder and click SynologyDrive. They should appear again. You'll get the icons and the right-click capabilities

<img width="659" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/76223f86-3110-427d-944f-503d44669889">

You'll want to sync it with different computers. In each one, you'll need to do this.

For the task name for a secondary mac, I would enter "lilmac"



## Sometimes problems with Synology NAS and raw files (.NEF)

## Other notes

One good thing to do is to keep the catalog on the local computer, but push the backups to the NAS. Make a backup folder in the nas.


Goal is to back up our local raw files to the nas - why not store them all permanently on the nas? - this is in addition to storing your raw files there...


Goals

1. Store all the raw files and make them accessible from anywhere
2. Have a backup of every Lightroom catalog


You're going to have all your raw files on the nas, not locally. The way to work with these is to create SMART previews. Not 1:1. Those don't replace smart.



If you have a mapped network drive, you'll need a VPN if you want to connect to it remotely. So make sure you have this worked out before you assume it's going to work when suddenly you connect to it from a new wifi network.


