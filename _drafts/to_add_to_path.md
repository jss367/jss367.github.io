$PATH is an environment variable used to lookup commands. 

When you run "ls" in a shell, for example, you actually run the /bin/ls program; the exact location may differ depending on your system configuration. This happens because /bin is in your $PATH.



to see where that command is, you can do `which ls`



(base) ➜  ~ which ls
ls: aliased to ls -G





(base) ➜  ~ where ls
ls: aliased to ls -G
/bin/ls



