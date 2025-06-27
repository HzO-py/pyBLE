import asyncio
import sys
import time
import datetime
import pandas as pd
from bleak import BleakScanner
from bleak import BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from bleak.backends.characteristic import BleakGATTCharacteristic

# In order to use this script on Linux, you need to bond/pair with the device first.
# The required BlueZ version is 5.43 or higher.
# You can use the command `bluetoothctl` to pair with the device.
# Here is a list of commands to pair with the device:
# 1. `bluetoothctl`
# 2. remove DA:72:C5:7B:79:89 // if the device is already paired or connected
# 3. // restart the webe band
# 4. // press both buttons on the band to enter pairing mode
# 5. scan on // to start scanning for devices
# 6. scan off // when you see the device in the list
# 7. connect DA:72:C5:7B:79:89 // to connect to the device
# 8. 555555 // when prompted for a passkey, enter 555555
# 9. disconnect // to disconnect from the device
# 10. exit // to exit the bluetoothctl command line interface
# 11. python linux_test.py // to run this script
# Above steps are required only once to pair with the device. After that, you can run this script directly.
UART_SERVICE_UUID = "2a3b580a-7263-4249-a647-e0af6f5966ab"
UART_RX_CHAR_UUID = "2a3b580b-7263-4249-a647-e0af6f5966ab"
UART_TX_CHAR_UUID = "2a3b580c-7263-4249-a647-e0af6f5966ab"

# BLEName = "We-Be 22f7"
BLEName = "We-Be 3d97"


total_bytes = 0
# SentCommand = False
flag_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
size = 150
time_gap = 0
out_counter = 0

data_received_tmp = []

def match_local_name(device: BLEDevice, advertisement_data: AdvertisementData):
    if advertisement_data.local_name and advertisement_data.local_name == BLEName:
        print(f'address: {device.address}, advertisement_data: {advertisement_data}')
        return True
    return False

def handle_disconnect(_: BleakClient):
    print("Device was disconnected, goodbye.")
    for task in asyncio.all_tasks():
        task.cancel()

def data_handler(data: bytearray):
    print("Received data:", data)
    print("Data length:", len(data))
    global out_counter
    global time_gap
    global flag_time
    global size
    global data_received_tmp
    

def handle_rx(_: BleakGATTCharacteristic, data: bytearray):
    global out_counter
    DecodedResponse = " ".join(f"0x{n:02x}" for n in data)
    print("received:", out_counter, " >> ", DecodedResponse)
    out_counter = out_counter + 1
    


async def send_command(client, command):
    command = bytearray(command)
    EncodedResponse = " ".join(f"0x{n:02x}" for n in command)
    print("Sent:", EncodedResponse)
    await client.write_gatt_char(UART_RX_CHAR_UUID, command)

async def main():
    # device = await BleakScanner.find_device_by_filter(match_local_name, timeout=10.0)
    device = await BleakScanner.find_device_by_name(BLEName, timeout=10.0)
    if device is None:
        print("Device not found. Please ensure the device is advertising.")
        sys.exit(1)
    print(f"Found device: {device.name} ({device.address})")

    async with BleakClient(device, disconnected_callback=handle_disconnect, pair=True) as client:
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)

        print("Connected to the device, syncronizing time...")
        await send_command(client, [5, 2, 0, 0])
        await asyncio.sleep(1.0)

        time_command = bytearray([5, 1, 0, 4])
        cur_time = time.time()
        for i in range(3, -1, -1):
            time_command.append(int(cur_time) >> (8 * i) & 0xff)
        # EncodedResponse = " ".join(f"0x{n:02x}" for n in time_command)
        # print("Sent:", EncodedResponse)
        # await client.write_gatt_char(UART_RX_CHAR_UUID, time_command)
        await send_command(client, time_command)
        await asyncio.sleep(5.0)
        print("Time synchronized, starting data profiling...")
        print("Press both buttons on the band to stop data profiling")

        print('Trying to start data profiling')
        await send_command(client, [1, 9, 0, 1, 1])

        await asyncio.Future()

        


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        # task is cancelled on disconnect, so we ignore this error
        print("Programm Calcelled.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, stopping data profiling...")
        # write data received list to file
        data_df = pd.DataFrame(data_received_tmp)
        output_filename = "data_received_tmp_c.csv"
        data_df.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")
    finally:
        print("Exiting the program.")
        sys.exit(0)
