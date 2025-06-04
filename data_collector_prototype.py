import asyncio
import sys
import time
import datetime
from bleak import BleakScanner
from bleak import BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
from bleak.backends.characteristic import BleakGATTCharacteristic


UART_SERVICE_UUID = "2a3b580a-7263-4249-a647-e0af6f5966ab"
UART_RX_CHAR_UUID = "2a3b580b-7263-4249-a647-e0af6f5966ab"
UART_TX_CHAR_UUID = "2a3b580c-7263-4249-a647-e0af6f5966ab"

BLEName = "We-Be 22f7"

total_bytes = 0
SentCommand = False
start_time = time.time()
flag_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
size = 150
time_gap = 0
out_counter = 0

def match_local_name(device: BLEDevice, advertisement_data: AdvertisementData):
    if advertisement_data.local_name and advertisement_data.local_name == BLEName:
        print(f'address: {device.address}, advertisement_data: {advertisement_data}')
        return True
    return False

def handle_disconnect(_: BleakClient):
    # global ifLog
    # global logFile
    print("Device was disconnected, goodbye.")
    for task in asyncio.all_tasks():
        task.cancel()

def base_data_handler(data):
    print("Base data handler not implemented yet.")
    print("Data received:", data)

def signal_data_handler(data):
    print("Signal data handler not implemented yet.")
    print("Data received:", data)

async def handle_rx(_: BleakGATTCharacteristic, data: bytearray):
    print("received:", data)
    global out_counter
    global start_time
    global SentCommand
    global total_bytes
    global df_live
    global lime_gap
    total_bytes += len(data)

    print(f"Total bytes received: {total_bytes}")
    if SentCommand or (data[0] == 0x04 and data[1] == 0x01):
        if SentCommand == False:
            start_time = time.time()
            SentCommand = True
    
    if data[0] == 0x01 and data[1] == 0x90 and len(data) > 42:
        base_data_handler(data)
    elif data[0] == 0x01 and data[1] == 0x91 and len(data) > 42:
        signal_data_handler(data)

    # elif data[0] == 0x01 and data[1] == 0x99 and len(data)== 6:
    #     df_live.to_csv("Logs/" + USER_NAME + ".csv", index=0)
    elif data[0] == 0x05 and data[1] == 0x02 and len(data) >=6:
        wearable_time = data[4]<<24|data[5]<<16|data[6]<<8|data[7]
        time_gap = int(time.time())-wearable_time
    else:
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

    async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)

        await send_command(client, [5, 2, 0, 0])

        time_command = bytearray([5, 1, 0, 4])
        cur_time = time.time()
        for i in range(3, -1, -1):
            time_command.append(int(cur_time) >> (8 * i) & 0xff)
        EncodedResponse = " ".join(f"0x{n:02x}" for n in time_command)
        print("Sent:", EncodedResponse)
        await client.write_gatt_char(UART_RX_CHAR_UUID, time_command)

        await asyncio.sleep(5.0)

        await send_command(client, [1, 9, 0, 1, 1])

        await asyncio.Future()

        


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        # task is cancelled on disconnect, so we ignore this error
        pass
