# OkCupid data
wget https://github.com/rudeboybert/JSE_OkCupid/raw/master/profiles.csv.zip
unzip profiles.csv
mv profiles.csv profiles.20120630.csv # to match the previous file name
rm profiles.csv.zip # clean up

# profanity data
if test -a 'profane.txt'; then
    echo "The profanity list has already been downloaded"
else
    wget http://search.cpan.org/CPAN/authors/id/T/TB/TBONE/Regexp-Profanity-US-1.4.tar.gz
    tar xvzf Regexp-Profanity-US-1.4.tar.gz
    rm Regexp-Profanity-US-1.4.tar.gz
    cat Regexp-Profanity-US-1.4/profane-definite.txt Regexp-Profanity-US-1.4/profane-ambiguous.txt | sed '/^$/d' > profane.txt
    rm -rf Regexp-Profanity-US-1.4
fi
