#!/bin/bash
# Check whether this is hopper or not.

case $HOSTNAME in
  (hopper1*) echo "hopper";;
  (hopper2*) echo "hopper";;
  (hop-amd-2*) echo "hopper";;
  (hop-amd-1*) echo "hopper";;
  (rorybox*) echo "dev";;
  (*)   echo "other";;
esac