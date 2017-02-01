#!/usr/bin/env bash
wget -O data/FB15k/fb15k.tgz https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz
tar -xzf data/FB15k/fb15k.tgz -C data/FB15k/
rm data/FB15k/fb15k.tgz