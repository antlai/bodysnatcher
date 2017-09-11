#!/bin/bash
echo 'Data in file /tmp/data.calib'
echo "[" > /tmp/data.calib
curl http://192.168.1.23:7090/points3D  >> /tmp/data.calib
echo "," >> /tmp/data.calib
read -p "Press enter to continue with 2"
curl http://192.168.1.23:7090/points3D  >> /tmp/data.calib
echo "," >> /tmp/data.calib
read -p "Press enter to continue with 3"
curl http://192.168.1.23:7090/points3D  >> /tmp/data.calib
echo "," >> /tmp/data.calib
read -p "Press enter to continue with 4"
curl http://192.168.1.23:7090/points3D  >> /tmp/data.calib
echo "," >> /tmp/data.calib
read -p "Press enter to continue with 5"
curl http://192.168.1.23:7090/points3D  >> /tmp/data.calib
echo "Done"
echo "]" >> /tmp/data.calib
