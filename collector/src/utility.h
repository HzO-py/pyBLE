#ifndef UTILITY_H
#define UTILITY_H

#include <stdint.h>

typedef struct {
    uint32_t timestamp;
    int32_t  temperature;
    uint16_t heart_rate_estim;
    uint8_t  hr_confidence;
    uint16_t rr_interbeat_interval;
    uint8_t  rr_confidence;
    uint16_t r_spo2;
    uint8_t  spo2_confidence;
    uint16_t spo2_estim;
    uint8_t  spo2_calc_percentage;
    uint8_t  whrm_suite_curr_opmode;
    uint8_t  spo2_low_sign_quality_flag;
    uint8_t  spo2_motion_flag;
    uint8_t  spo2_low_pi_flag;
    uint8_t  spo2_unreliable_r_flag;
    uint8_t  spo2_state;
    uint8_t  skin_contact_state;
    uint8_t  activity_class;
    uint32_t walk_steps;
    uint32_t run_steps;
    uint32_t kcal;
    uint32_t cadence;
    uint8_t  event;

    uint32_t grn_count;
    uint32_t ir_cnt;
    uint32_t red_cnt;
    uint32_t grn2_cnt;
    int16_t  accel_x;
    int16_t  accel_y;
    int16_t  accel_z;
    uint16_t gsr;

} ParsedData;


typedef enum {
    RESULT_DATA_PROCESSED = 0, // Unified success code
    RESULT_STOP_CMD_RECEIVED = 1,
    RESULT_TIME_SYNC_PROCESSED = 2,
    RESULT_PARSING_ERROR = -1,
    RESULT_INSUFFICIENT_DATA = -2,
    RESULT_SHORT_DATA_ERROR = -3,
    RESULT_UNKNOWN_PACKET = -4,
} HandlerResult;


HandlerResult data_handler(const unsigned char *data, int data_len, ParsedData *output_array, int *output_count);


#endif
