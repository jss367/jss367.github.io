


If you just don't want a warning, you can ignore it like so:

`source FILE 2> /dev/null`


backticks and single quotes

Single quotes won't interpolate anything, but double quotes will

this answer has good info: https://stackoverflow.com/a/42082956/2514130


brackets are like the test command. They run a statement


This works on my mac, but not linux:

if [ "$(uname -m)" == "arm64" ]; then
    echo "ARM64 detected"
elif [ "$(uname -m)" == "x86_64" ]; then
    echo "x86_64 detected"
else
    echo "ERROR! OS not recognized"
fi


This works on both:

if [ "$(uname -m)" = "arm64" ]; then
    echo "ARM64 detected"
elif [ "$(uname -m)" = "x86_64" ]; then
    echo "x86_64 detected"
else
    echo "ERROR! OS not recognized"
fi
