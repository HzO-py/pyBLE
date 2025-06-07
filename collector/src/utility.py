import ctypes
import os
import sys

class ParsedData(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_uint32),
        ("temperature", ctypes.c_int32),
        ("heart_rate_estim", ctypes.c_uint16),
        ("hr_confidence", ctypes.c_uint8),
        ("rr_interbeat_interval", ctypes.c_uint16),
        ("rr_confidence", ctypes.c_uint8),
        ("r_spo2", ctypes.c_uint16),
        ("spo2_confidence", ctypes.c_uint8),
        ("spo2_estim", ctypes.c_uint16),
        ("spo2_calc_percentage", ctypes.c_uint8),
        ("whrm_suite_curr_opmode", ctypes.c_uint8),
        ("spo2_low_sign_quality_flag", ctypes.c_uint8),
        ("spo2_motion_flag", ctypes.c_uint8),
        ("spo2_low_pi_flag", ctypes.c_uint8),
        ("spo2_unreliable_r_flag", ctypes.c_uint8),
        ("spo2_state", ctypes.c_uint8),
        ("skin_contact_state", ctypes.c_uint8),
        ("activity_class", ctypes.c_uint8),
        ("walk_steps", ctypes.c_uint32),
        ("run_steps", ctypes.c_uint32),
        ("kcal", ctypes.c_uint32),
        ("cadence", ctypes.c_uint32),
        ("event", ctypes.c_uint8),
        ("grn_count", ctypes.c_uint32),
        ("ir_cnt", ctypes.c_uint32),
        ("red_cnt", ctypes.c_uint32),
        ("grn2_cnt", ctypes.c_uint32),
        ("accel_x", ctypes.c_int16),
        ("accel_y", ctypes.c_int16),
        ("accel_z", ctypes.c_int16),
        ("gsr", ctypes.c_uint16),
    ]

class SignalData(ctypes.Structure):
    _fields_ = [
        ("grn_count", ctypes.c_uint32),
        ("ir_cnt", ctypes.c_uint32),
        ("red_cnt", ctypes.c_uint32),
        ("grn2_cnt", ctypes.c_uint32),
        ("accel_x", ctypes.c_int16),
        ("accel_y", ctypes.c_int16),
        ("accel_z", ctypes.c_int16),
        ("gsr", ctypes.c_uint16),
    ]

def load_utility_library():
    LIB_NAME = 'utility.so' if sys.platform != 'win32' else 'utility.dll'

    try:
        lib_path = os.path.join(os.path.dirname(__file__), 'output', LIB_NAME)
        return ctypes.CDLL(lib_path)
    except OSError as e:
        print(f"Error: Could not load the shared library from '{lib_path}'")
        print("Please make sure you have run 'make' to compile the C code.")
        print(f"Details: {e}")
        sys.exit(1)

utility_lib = load_utility_library()

def setup_prototypes():
    utility_lib.base_data_handler.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(ParsedData),
        ctypes.POINTER(ctypes.c_int)
    ]
    utility_lib.base_data_handler.restype = ctypes.c_int

    utility_lib.signal_data_handler.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
        ctypes.POINTER(SignalData),
        ctypes.POINTER(ctypes.c_int)
    ]
    utility_lib.signal_data_handler.restype = ctypes.c_int

setup_prototypes()

def c_base_data_handler_wrapper(data: bytearray):
    data_len = len(data)
    c_data_buffer = (ctypes.c_ubyte * data_len).from_buffer(data)
    output_array_type = ParsedData * 7
    output_array = output_array_type()
    output_count = ctypes.c_int(0)

    ret = utility_lib.base_data_handler(c_data_buffer, data_len, output_array, ctypes.byref(output_count))

    results = []
    if ret == 0:
        print(f"C base_data_handler processed {output_count.value} records.")
        for i in range(output_count.value):
            data_dict = {field_def[0]: getattr(output_array[i], field_def[0]) for field_def in ParsedData._fields_}
            results.append(data_dict)
    else:
        print(f"C base_data_handler returned an error: {ret}")
    return results


def c_signal_data_handler_wrapper(data: bytearray):
    """
    Python wrapper for the C signal_data_handler function.
    Takes a bytearray and returns a list of dictionaries.
    """
    data_len = len(data)
    c_data_buffer = (ctypes.c_ubyte * data_len).from_buffer(data)
    output_array_type = SignalData * 9
    output_array = output_array_type()
    output_count = ctypes.c_int(0)

    ret = utility_lib.signal_data_handler(c_data_buffer, data_len, output_array, ctypes.byref(output_count))

    results = []
    if ret == 0:
        print(f"C signal_data_handler processed {output_count.value} records.")
        for i in range(output_count.value):
            data_dict = {field_def[0]: getattr(output_array[i], field_def[0]) for field_def in SignalData._fields_}
            results.append(data_dict)
    else:
        print(f"C signal_data_handler returned an error: {ret}")
    return results