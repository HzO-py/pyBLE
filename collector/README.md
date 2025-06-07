# Collect Webe Data with Collector
## Overview
This directory contains the code for collecting data from Webe. The BLE communication is handled in python scripts and the data processing is don in C.
## Requirements
- Bleak == 0.22.3
- GCC >= 11.2.0

## Usage
### OS X - CoreBluetooth
OS X support via Core Bluetooth API, from OS X 10.13.
Just run the sample_collector.py script and make sure the Webe band is on.
The passkey input through a pop up window.
### Linux - BlueZ >= 5.55
To use on Linux, due to the limitation of BlueZ, manual pairing is required.
If you are paring the webe band with linux device for the first time, follow the instruction below.
```
<!-- For the first time pairing -->
$ bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# default-agent
<!-- Hold both buttons on the Webe Band to enter parining mode -->
[bluetooth]# scan on
<!-- Wait until you see the name of your webe device -->
[bluetooth]# scan off
[bluetooth]# connect <MAC_ADDRESS>
<!-- Enter passkey here -->
[bluetooth]# disconnect
[bluetooth]# quit
$ python3 sample_collector.py
```
In some cases, you may need to pair the device again. Follow the instruction below to re-pair the device.
```
<!-- For re-pairing -->
$ bluetoothctl
[bluetooth]# remove <MAC_ADDRESS>
[bluetooth]# quit
$ bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# default-agent
<!-- Turn Webe Band off and on again -->
<!-- Hold both buttons on the Webe Band to enter parining mode -->
[bluetooth]# scan on
<!-- Wait until you see the name of your webe device -->
[bluetooth]# scan off
[bluetooth]# connect <MAC_ADDRESS>
<!-- Enter passkey here -->
[bluetooth]# disconnect
[bluetooth]# quit
$ python3 sample_collector.py
```