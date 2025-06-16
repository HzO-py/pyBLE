#include "utility.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>


int base_data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count);
int signal_data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count);


static uint16_t encryption_weights[5] = {0x1234, 0xABCD, 0x1F1F, 0x0F0F, 0xAAAA};

// Feistel encryption key
static uint32_t encryption_key = 0xDEADBEEF;

// Simple PRNG (xorshift) to shuffle with key as seed
uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Initialize S-box with key as seed
void init_sbox(uint8_t sbox[256], uint32_t key) {
    for (int i = 0; i < 256; i++)
        sbox[i] = i;

    uint32_t state = key;
    for (int i = 255; i > 0; i--) {
        uint32_t r = xorshift32(&state) % (i + 1);
        uint8_t tmp = sbox[i];
        sbox[i] = sbox[r];
        sbox[r] = tmp;
    }
}

// Round function F using S-box and round key
uint8_t feistel_round(uint8_t half, uint8_t round_key, uint8_t sbox[256]) {
    return sbox[half ^ round_key];
}

// Encrypt 16-bit value with 4-round Feistel cipher
uint16_t feistel_encrypt_16bit(uint16_t value, uint32_t master_key) {
    uint8_t sbox[256];
    init_sbox(sbox, master_key);

    uint8_t round_keys[4];
    uint32_t rk_state = master_key;
    for (int i = 0; i < 4; i++)
        round_keys[i] = xorshift32(&rk_state) & 0xFF;

    uint8_t L = (value >> 8) & 0xFF;
    uint8_t R = value & 0xFF;

    for (uint8_t i = 0; i < 4; i++) {
        uint8_t temp = R;
        R = L ^ feistel_round(R, round_keys[i], sbox);
        L = temp;
    }

    return ((uint16_t)L << 8) | R;
}

// Compute weighted checksum of data (2 bytes at a time), result is 16-bit
uint16_t weighted_checksum_16bit(const uint8_t *data, size_t len, const uint16_t *weights, size_t num_weights) {
    uint32_t sum = 0;

    for (size_t i = 0; i + 1 < len; i += 2) {
        uint16_t word = ((uint16_t)data[i] << 8) | data[i + 1];
        sum += word * weights[(i / 2) % num_weights];
    }

    return (uint16_t)(sum & 0xFFFF);  // return lower 16 bits
}

bool encryption_check(const uint8_t *data, uint8_t input_low_byte, uint8_t input_high_byte) {
    uint16_t encrypted_checksum = feistel_encrypt_16bit(weighted_checksum_16bit(data, 180, encryption_weights, 5), encryption_key);
    uint8_t low_byte = encrypted_checksum & 0xFF;
    uint8_t high_byte = (encrypted_checksum >> 8) & 0xFF;
    if (low_byte != input_low_byte || high_byte != input_high_byte) {
        printf("Encryption check failed: expected %02X%02X, got %02X%02X\n", high_byte, low_byte, input_high_byte, input_low_byte);
        return false;
    }
    return true; // Placeholder for actual encryption check logic
}

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
        if (encryption_check(&data[6], data[5], data[4])) {
            // Encryption check passed, proceed with data handling
            return (HandlerResult)base_data_handler(data, data_len, (ParsedData*)output_array, output_count);
        } else {
            printf("C Handler: Encryption check failed\n");
            return RESULT_PARSING_ERROR; // -1
        }
    }
    if (data[0] == 0x01 && data[1] == 0x91) {
        if (encryption_check(&data[6], data[5], data[4])) {
            // Encryption check passed, proceed with signal data handling
            return (HandlerResult)signal_data_handler(data, data_len, (ParsedData*)output_array, output_count);
        } else {
            printf("C Handler: Encryption check failed\n");
            return RESULT_PARSING_ERROR; // -1
        }
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