#!/bin/bash
export PYTHONHASHSEED=0
echo "START SUPERLINK"
flower-superlink --insecure &
sleep 5
echo "START CLIENT1 APP"
flower-client-app client1:app --insecure --dir "./app1/custom" &
echo "START CLIENT2 APP"
flower-client-app client2:app --insecure --dir "./app2/custom" &
sleep 5
echo "START SERVER APP"
flower-server-app server:app --insecure --dir "./server/custom"
