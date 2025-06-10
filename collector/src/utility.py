import ctypes
import os
import sys
from enum import IntEnum

class ParsedData(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_int32), ("temperature", ctypes.c_int32),
        ("heart_rate_estim", ctypes.c_int16), ("hr_confidence", ctypes.c_int8),
        ("rr_interbeat_interval", ctypes.c_int16), ("rr_confidence", ctypes.c_int8),
        ("r_spo2", ctypes.c_int16), ("spo2_confidence", ctypes.c_int8),
        ("spo2_estim", ctypes.c_int16), ("spo2_calc_percentage", ctypes.c_int8),
        ("whrm_suite_curr_opmode", ctypes.c_int8), ("spo2_low_sign_quality_flag", ctypes.c_int8),
        ("spo2_motion_flag", ctypes.c_int8), ("spo2_low_pi_flag", ctypes.c_int8),
        ("spo2_unreliable_r_flag", ctypes.c_int8), ("spo2_state", ctypes.c_int8),
        ("skin_contact_state", ctypes.c_int8), ("activity_class", ctypes.c_int8),
        ("walk_steps", ctypes.c_int32), ("run_steps", ctypes.c_int32),
        ("kcal", ctypes.c_int32), ("cadence", ctypes.c_int32),
        ("event", ctypes.c_int8), ("grn_count", ctypes.c_int32),
        ("ir_cnt", ctypes.c_int32), ("red_cnt", ctypes.c_int32),
        ("grn2_cnt", ctypes.c_int32), ("accel_x", ctypes.c_int16),
        ("accel_y", ctypes.c_int16), ("accel_z", ctypes.c_int16),
        ("gsr", ctypes.c_int16),
    ]

class HandlerResult(IntEnum):
    RESULT_DATA_PROCESSED = 0
    RESULT_STOP_CMD_RECEIVED = 1
    RESULT_TIME_SYNC_PROCESSED = 2
    RESULT_PARSING_ERROR = -1
    RESULT_INSUFFICIENT_DATA = -2
    RESULT_SHORT_DATA_ERROR = -3
    RESULT_UNKNOWN_PACKET = -4

def load_and_setup():
    LIB_NAME = 'utility.so'
    if sys.platform.startswith('win'):
        LIB_NAME = 'utility.dll'
    try:
        lib_path = os.path.join(os.path.dirname(__file__), 'output', LIB_NAME)
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"Error: Could not load shared library from '{lib_path}'", file=sys.stderr)
        sys.exit(1)

    lib.data_handler.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
    ]
    lib.data_handler.restype = ctypes.c_int
    return lib

utility_lib = load_and_setup()

def c_data_handler_wrapper(data: bytearray):
    data_len = len(data)
    c_data_buffer = (ctypes.c_ubyte * data_len).from_buffer(data)
    
    MAX_RECORDS = 9
    output_buffer = (ParsedData * MAX_RECORDS)()
    output_count = ctypes.c_int(0)

    ret_code = utility_lib.data_handler(
        c_data_buffer, data_len, 
        ctypes.byref(output_buffer), ctypes.byref(output_count)
    )
    
    status = HandlerResult(ret_code)
    results = []

    if status == HandlerResult.RESULT_DATA_PROCESSED:
        for i in range(output_count.value):
            data_dict = {field[0]: getattr(output_buffer[i], field[0]) for field in ParsedData._fields_}
            results.append(data_dict)
            
    return status, results
