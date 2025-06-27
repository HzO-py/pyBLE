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
# BLEName = "We-Be 3d97"
BLEName = "We-Be 17f6"


total_bytes = 0
# SentCommand = False
flag_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
size = 150
time_gap = 0
out_counter = 0

data_received_tmp = []


def base_data_handler(data):
    global df_live
    EncodedReceived = " ".join(f"0x{n:02x}" for n in data)
    print(EncodedReceived)

    base_index = 6

    # assert data[base_index+0] == 0xff
    # assert data[base_index+1] == 0xee

    data_dict = {
        'timestamp': data[base_index + 2]<<24|data[base_index + 3]<<16|data[base_index + 4]<<8|data[base_index + 5],
        'temperature': data[base_index + 6]<<24|data[base_index + 7]<<16|data[base_index + 8]<<8|data[base_index + 9],
        'heart_rate_estim': data[base_index + 10]<<8|data[base_index + 11],
        'hr_confidence' : data[base_index + 12],
        'rr_interbeat_interval': data[base_index + 13]<<8|data[base_index + 14],
        'rr_confidence' : data[base_index + 15],
        'r_spo2': data[base_index + 16]<<8|data[base_index + 17],
        'spo2_confidence': data[base_index + 18],
        'spo2_estim': data[base_index + 19]<<8|data[base_index + 20],
        'spo2_calc_percentage': data[base_index + 21],
        'whrm_suite_curr_opmode': 0,
        'spo2_low_sign_quality_flag': data[base_index +22] >> 7 & 0x01,
        'spo2_motion_flag':data[base_index +22] >> 6 & 0x01,
        'spo2_low_pi_flag': data[base_index +22] >> 5 & 0x01,
        'spo2_unreliable_r_flag': data[base_index +22] >> 4 & 0x01,
        'spo2_state': data[base_index +22] & 0x03,
        'skin_contact_state':data[base_index + 23]&0x02,
        'activity_class': data[base_index + 23]>>4,
        'walk_steps':data[base_index + 24]<<24|data[base_index + 25]<<16|data[base_index + 26]<<8|data[base_index + 27],
        'run_steps': data[base_index + 28]<<24|data[base_index + 29]<<16|data[base_index + 30]<<8|data[base_index + 31],
        'kcal': data[base_index + 32]<<24|data[base_index + 33]<<16|data[base_index + 34]<<8|data[base_index + 35],
        'cadence': data[base_index + 36]<<24|data[base_index + 37]<<16|data[base_index + 38]<<8|data[base_index + 39],
        'event': (data[base_index + 23]>>2) & 0x03,

        'grn_count': data[base_index + 40] << 16 | data[base_index + 41] << 8 | data[base_index + 42],
        'irCnt': data[base_index + 43] << 16 | data[base_index + 44] << 8 | data[base_index + 45],
        'redCnt': data[base_index + 46] << 16 | data[base_index + 47] << 8 | data[base_index + 48],
        'grn2Cnt': data[base_index + 49] << 16 | data[base_index + 50] << 8 | data[base_index + 51],

        'accelx': data[base_index + 52] << 8 | data[base_index + 53],
        'accely': data[base_index + 54] << 8 | data[base_index + 55],
        'accelz': data[base_index + 56] << 8 | data[base_index + 57],

        'GSR': data[base_index + 58] << 8 | data[base_index + 59]
    }
    data_received_tmp.append(data_dict)
    print(f"green_count: {data_dict['grn_count']}, irCnt: {data_dict['irCnt']}, redCnt: {data_dict['redCnt']}, grn2Cnt: {data_dict['grn2Cnt']}")

    base_index = base_index + 60
    for i in range(6):
        data_dict = {
            'grn_count': data[base_index + 0]<<16|data[base_index + 1]<<8|data[base_index + 2],
            'irCnt': data[base_index + 3] << 16 | data[base_index + 4] << 8 | data[base_index + 5],
            'redCnt': data[base_index + 6] << 16 | data[base_index + 7] << 8 | data[base_index + 8],
            'grn2Cnt': data[base_index + 9] << 16 | data[base_index + 10] << 8 | data[base_index + 11],

            'accelx': data[base_index + 12] << 8 | data[base_index + 13],
            'accely': data[base_index + 14] << 8 | data[base_index + 15],
            'accelz': data[base_index + 16] << 8 | data[base_index + 17],

            'GSR': data[base_index+18] << 8 | data[base_index+19]
        }
        data_received_tmp.append(data_dict)
        print(f"green_count: {data_dict['grn_count']}, irCnt: {data_dict['irCnt']}, redCnt: {data_dict['redCnt']}, grn2Cnt: {data_dict['grn2Cnt']}")

        base_index += 20

def signal_data_handler(data):
    global df_live

    base_index = 6
    for i in range(9):
        data_dict = {
            'grn_count': data[base_index + 0] << 16 | data[base_index + 1] << 8 | data[base_index + 2],
            'irCnt': data[base_index + 3] << 16 | data[base_index + 4] << 8 | data[base_index + 5],
            'redCnt': data[base_index + 6] << 16 | data[base_index + 7] << 8 | data[base_index + 8],
            'grn2Cnt': data[base_index + 9] << 16 | data[base_index + 10] << 8 | data[base_index + 11],

            'accelx': data[base_index + 12] << 8 | data[base_index + 13],
            'accely': data[base_index + 14] << 8 | data[base_index + 15],
            'accelz': data[base_index + 16] << 8 | data[base_index + 17],

            'GSR': data[base_index + 18] << 8 | data[base_index + 19]
        }
        data_received_tmp.append(data_dict)
    print(f"green_count: {data_dict['grn_count']}, irCnt: {data_dict['irCnt']}, redCnt: {data_dict['redCnt']}, grn2Cnt: {data_dict['grn2Cnt']}")

    base_index = base_index + 20


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
    global total_bytes
    global out_counter
    if data[0] == 0x01 and data[1] == 0x92 and len(data) > 42:
        base_data_handler(data)
    elif data[0] == 0x01 and data[1] == 0x93 and len(data) > 42:
        signal_data_handler(data)

    elif data[0] == 0x01 and data[1] == 0x21 and data[3] == 0x00 and data[4] == 0x01:
        data_df = pd.DataFrame(data_received_tmp)
        output_filename = "data_received_tmp_c.csv"
        data_df.to_csv(output_filename, index=False)
        print(f"Data saved to {output_filename}")
        print("Manually stopped data profiling")
        for task in asyncio.all_tasks():
            task.cancel()
    elif data[0] == 0x05 and data[1] == 0x02 and len(data) >=6:
        wearable_time = data[4]<<24|data[5]<<16|data[6]<<8|data[7]
        time_gap = int(time.time())-wearable_time
        print(f"Time synchronized, time gap: {time_gap} seconds")
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
        await send_command(client, [1, 9, 0, 1, 2, 1])
        await asyncio.sleep(10.0)
        await send_command(client, [1, 23, 17, 3, 255, 0, 0])
        await asyncio.sleep(5.0)
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
