#!/usr/bin/bash
wget https://github.com/Virinas-code/Crocrodile/archive/refs/tags/v1.1.0.tar.gz
tar -zxvf v1.1.0.tar.gz
cd Crocrodile-1.1.0
read -p "Token : "
echo "$REPLY" > lichess.token
clear
bash start

