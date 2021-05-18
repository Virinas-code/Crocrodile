#!/usr/bin/bash
wget https://github.com/Virinas-code/Crocrodile/archive/refs/tags/v2.0.0.tar.gz
tar -zxvf v2.0.0.tar.gz
cd Crocrodile-2.0.0
read -p "Token : "
echo "$REPLY" > lichess.token
clear
bash start
