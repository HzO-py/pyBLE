#include "utility.h"
#include <stdio.h>
#include <string.h>


int base_data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count);
int signal_data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count);


int data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count) {
    *output_count = 0; // Initialize output count to 0
    if (data_len < 4) return RESULT_INSUFFICIENT_DATA; // -2

    if (data_len < 186 ) {
        if (data[0] == 0x05 && data[1] == 0x02 && data_len >= 8) {
            return RESULT_TIME_SYNC_PROCESSED; // 2
        }
        if (data[0] == 0x01 && data[1] == 0x21 && data[2] == 0x00 && data[3] == 0x01) {
            return RESULT_STOP_CMD_RECEIVED; // 1
        }
        if (data[0] == 0x01 && data[1] == 0x99) {
            return RESULT_SHORT_DATA_ERROR; // -3
        }
    }


    if (data[0] == 0x01 && data[1] == 0x90) {
        return (HandlerResult)base_data_handler(data, data_len, (ParsedData*)output_array, output_count);
    }
    if (data[0] == 0x01 && data[1] == 0x91) {
        return (HandlerResult)signal_data_handler(data, data_len, (ParsedData*)output_array, output_count);
    }
    
    printf("C Handler: Unknown packet received\n");
    return RESULT_UNKNOWN_PACKET; // -4
}



int base_data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count) {
    const int BASE_INDEX = 6;
    const int EXPECTED_MIN_LEN = BASE_INDEX + 60 + (6 * 20); // 186 bytes

    // --- Pre-flight Checks ---
    if (data_len < EXPECTED_MIN_LEN) { *output_count = 0; return -2; }
    if (data[BASE_INDEX] != 0xff || data[BASE_INDEX + 1] != 0xee) { *output_count = 0; return -1; }

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

        // Initialize all non-signal fields to -1
        p->timestamp = -1; p->temperature = -1; p->heart_rate_estim = -1;
        p->hr_confidence = -1; p->rr_interbeat_interval = -1; p->rr_confidence = -1;
        p->r_spo2 = -1; p->spo2_confidence = -1; p->spo2_estim = -1;
        p->spo2_calc_percentage = -1; p->whrm_suite_curr_opmode = -1;
        p->spo2_low_sign_quality_flag = -1; p->spo2_motion_flag = -1;
        p->spo2_low_pi_flag = -1; p->spo2_unreliable_r_flag = -1;
        p->spo2_state = -1; p->skin_contact_state = -1; p->activity_class = -1;
        p->walk_steps = -1; p->run_steps = -1; p->kcal = -1; p->cadence = -1; p->event = -1;

        // Populate fields that are present in the signal packet
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

int signal_data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count) {
    const int BASE_INDEX = 6;
    const int EXPECTED_MIN_LEN = BASE_INDEX + (9 * 20); // 186 bytes

    // --- Pre-flight Check for data length ---
    if (data_len < EXPECTED_MIN_LEN) { *output_count = 0; return -2; }

    // Clear the output array to ensure all fields start at 0
    memset(output_array, 0, sizeof(ParsedData) * 9);

    int current_index = BASE_INDEX;
    *output_count = 0;

    // --- Parse 9 Blocks (20 bytes each) ---
    for (int i = 0; i < 9; i++) {
        ParsedData *p = &output_array[i]; // Pointer to the current element
        
        // Initialize all non-signal fields to -1
        p->timestamp = -1; p->temperature = -1; p->heart_rate_estim = -1;
        p->hr_confidence = -1; p->rr_interbeat_interval = -1; p->rr_confidence = -1;
        p->r_spo2 = -1; p->spo2_confidence = -1; p->spo2_estim = -1;
        p->spo2_calc_percentage = -1; p->whrm_suite_curr_opmode = -1;
        p->spo2_low_sign_quality_flag = -1; p->spo2_motion_flag = -1;
        p->spo2_low_pi_flag = -1; p->spo2_unreliable_r_flag = -1;
        p->spo2_state = -1; p->skin_contact_state = -1; p->activity_class = -1;
        p->walk_steps = -1; p->run_steps = -1; p->kcal = -1; p->cadence = -1; p->event = -1;

        // Populate fields that are present in the signal packet
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