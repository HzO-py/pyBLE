#include "utility.h"
#include <stdio.h>
#include <string.h>

int base_data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count) {
    const int BASE_INDEX = 6;
    const int EXPECTED_MIN_LEN = BASE_INDEX + 60 + (6 * 20); // 186 bytes

    // --- Pre-flight Checks ---
    if (data_len < EXPECTED_MIN_LEN) {
        *output_count = 0;
        return -2; // Error: Insufficient data length
    }

    if (data[BASE_INDEX + 0] != 0xff || data[BASE_INDEX + 1] != 0xee) {
        *output_count = 0;
        return -1; // Error: Invalid header
    }

    // Clear the output array to ensure all fields start at 0
    memset(output_array, 0, sizeof(ParsedData) * 7);

    // --- Parse Primary Block (First 60 bytes after header) ---
    int current_index = BASE_INDEX;
    ParsedData *p = &output_array[0]; // Pointer to the first element

    p->timestamp = (data[current_index + 2] << 24) | (data[current_index + 3] << 16) | (data[current_index + 4] << 8) | data[current_index + 5];
    p->temperature = (data[current_index + 6] << 24) | (data[current_index + 7] << 16) | (data[current_index + 8] << 8) | data[current_index + 9];
    p->heart_rate_estim = (data[current_index + 10] << 8) | data[current_index + 11];
    p->hr_confidence = data[current_index + 12];
    p->rr_interbeat_interval = (data[current_index + 13] << 8) | data[current_index + 14];
    p->rr_confidence = data[current_index + 15];
    p->r_spo2 = (data[current_index + 16] << 8) | data[current_index + 17];
    p->spo2_confidence = data[current_index + 18];
    p->spo2_estim = (data[current_index + 19] << 8) | data[current_index + 20];
    p->spo2_calc_percentage = data[current_index + 21];

    p->spo2_low_sign_quality_flag = (data[current_index + 22] >> 7) & 0x01;
    p->spo2_motion_flag = (data[current_index + 22] >> 6) & 0x01;
    p->spo2_low_pi_flag = (data[current_index + 22] >> 5) & 0x01;
    p->spo2_unreliable_r_flag = (data[current_index + 22] >> 4) & 0x01;
    p->spo2_state = data[current_index + 22] & 0x03;
    p->skin_contact_state = data[current_index + 23] & 0x02;
    p->activity_class = data[current_index + 23] >> 4;
    p->event = (data[current_index + 23] >> 2) & 0x03;

    p->walk_steps = (data[current_index + 24] << 24) | (data[current_index + 25] << 16) | (data[current_index + 26] << 8) | data[current_index + 27];
    p->run_steps = (data[current_index + 28] << 24) | (data[current_index + 29] << 16) | (data[current_index + 30] << 8) | data[current_index + 31];
    p->kcal = (data[current_index + 32] << 24) | (data[current_index + 33] << 16) | (data[current_index + 34] << 8) | data[current_index + 35];
    p->cadence = (data[current_index + 36] << 24) | (data[current_index + 37] << 16) | (data[current_index + 38] << 8) | data[current_index + 39];

    p->grn_count = (data[current_index + 40] << 16) | (data[current_index + 41] << 8) | data[current_index + 42];
    p->ir_cnt = (data[current_index + 43] << 16) | (data[current_index + 44] << 8) | data[current_index + 45];
    p->red_cnt = (data[current_index + 46] << 16) | (data[current_index + 47] << 8) | data[current_index + 48];
    p->grn2_cnt = (data[current_index + 49] << 16) | (data[current_index + 50] << 8) | data[current_index + 51];

    p->accel_x = (data[current_index + 52] << 8) | data[current_index + 53];
    p->accel_y = (data[current_index + 54] << 8) | data[current_index + 55];
    p->accel_z = (data[current_index + 56] << 8) | data[current_index + 57];
    p->gsr = (data[current_index + 58] << 8) | data[current_index + 59];

    current_index += 60;
    *output_count = 1;

    // --- Parse Subsequent 6 Blocks (20 bytes each) ---
    for (int i = 0; i < 6; i++) {
        p = &output_array[i + 1]; // Pointer to the next element in the output array

        p->grn_count = (data[current_index + 0] << 16) | (data[current_index + 1] << 8) | data[current_index + 2];
        p->ir_cnt    = (data[current_index + 3] << 16) | (data[current_index + 4] << 8) | data[current_index + 5];
        p->red_cnt   = (data[current_index + 6] << 16) | (data[current_index + 7] << 8) | data[current_index + 8];
        p->grn2_cnt  = (data[current_index + 9] << 16) | (data[current_index + 10] << 8) | data[current_index + 11];

        p->accel_x = (data[current_index + 12] << 8) | data[current_index + 13];
        p->accel_y = (data[current_index + 14] << 8) | data[current_index + 15];
        p->accel_z = (data[current_index + 16] << 8) | data[current_index + 17];
        p->gsr     = (data[current_index + 18] << 8) | data[current_index + 19];

        current_index += 20;
        (*output_count)++;
    }

    return 0; // Success
}

int signal_data_handler(const unsigned char *data, int data_len, SignalData *output_array, int *output_count) {
    const int BASE_INDEX = 6;
    const int EXPECTED_MIN_LEN = BASE_INDEX + (9 * 20); // 186 bytes

    // --- Pre-flight Check for data length ---
    if (data_len < EXPECTED_MIN_LEN) {
        *output_count = 0;
        return -2; // Error: Insufficient data length
    }

    // Clear the output array to ensure all fields start at 0
    memset(output_array, 0, sizeof(SignalData) * 9);

    int current_index = BASE_INDEX;
    *output_count = 0;

    // --- Parse 9 Blocks (20 bytes each) ---
    for (int i = 0; i < 9; i++) {
        SignalData *p = &output_array[i]; // Pointer to the current element

        p->grn_count = (data[current_index + 0] << 16) | (data[current_index + 1] << 8) | data[current_index + 2];
        p->ir_cnt    = (data[current_index + 3] << 16) | (data[current_index + 4] << 8) | data[current_index + 5];
        p->red_cnt   = (data[current_index + 6] << 16) | (data[current_index + 7] << 8) | data[current_index + 8];
        p->grn2_cnt  = (data[current_index + 9] << 16) | (data[current_index + 10] << 8) | data[current_index + 11];

        p->accel_x = (data[current_index + 12] << 8) | data[current_index + 13];
        p->accel_y = (data[current_index + 14] << 8) | data[current_index + 15];
        p->accel_z = (data[current_index + 16] << 8) | data[current_index + 17];
        p->gsr     = (data[current_index + 18] << 8) | data[current_index + 19];

        current_index += 20;
        (*output_count)++;
    }

    return 0; // Success
}