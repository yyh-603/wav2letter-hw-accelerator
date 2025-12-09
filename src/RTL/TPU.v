`include "./src/RTL/PE.v"
`include "./src/RTL/systolic_array.v"

module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    input_offset,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);

input clk;
input rst_n;
input            in_valid;
input [7:0]      K;
input [7:0]      M;
input [7:0]      N;
input signed [8:0]      input_offset;
output  reg      busy;

output           A_wr_en;
output [15:0]    A_index;
output [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output [15:0]    B_index;
output [31:0]    B_data_in;
input  [31:0]    B_data_out;

output           C_wr_en;
output [15:0]    C_index;
output [127:0]   C_data_in;
input  [127:0]   C_data_out;

//* Implement your design here

reg [7:0] K_reg, M_reg, N_reg;
reg [15:0] index_r, index_c, C_shift_1, C_shift_2;
reg [15:0] A_index_base, B_index_base, C_index_base;
reg signed [31:0] A_buf [0:3];
reg signed [31:0] B_buf [0:3];
reg signed [127:0] C_buf [0:3];
wire [511:0] C_output;
wire sa_busy;
reg sa_in_valid, sa_clear;

localparam IDLE       = 4'd0,
           SA_CLEAR_1 = 4'd1,
           SA_CLEAR_2 = 4'd2,
           READ_ID    = 4'd3,
           READ_DATA  = 4'd4,
           SA_START   = 4'd5,
           SA_WAIT_1  = 4'd6,
           SA_WAIT_2  = 4'd7,
           C_SAVE     = 4'd8,
           NEXT_LOOP  = 4'd9,
           RESET      = 4'd10;



// typedef enum logic [3:0] {IDLE, // 0 
//                           SA_CLEAR_1, SA_CLEAR_2, // 1, 2
//                           READ_ID, READ_DATA, // 3, 4
//                           SA_START, SA_WAIT_1, SA_WAIT_2, // 7, 8, 9
//                           C_SAVE, // A
//                           NEXT_LOOP, RESET} state_t; // B, C
reg [3:0] s_cur, s_nxt;
reg [1:0] s_cnt;

systolic_array m_systolic_array(
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(sa_in_valid),
    .clear(sa_clear),
    .row_in({A_buf[0], A_buf[1], A_buf[2], A_buf[3]}),
    .col_in({B_buf[0], B_buf[1], B_buf[2], B_buf[3]}),
    .input_offset(input_offset),
    .data_out(C_output),
    .busy(sa_busy)
);

assign A_wr_en = 0;
assign A_index = index_r * K_reg + A_index_base + s_cnt;
assign A_data_in = 0;

assign B_wr_en = 0;
assign B_index = index_c * K_reg + B_index_base + s_cnt;
assign B_data_in = 0;

assign C_wr_en = (s_cur == C_SAVE);
assign C_index = C_shift_1 + C_shift_2 + C_index_base + s_cnt;
assign C_data_in = 
    (s_cnt == 0) ? C_buf[0] : 
    (s_cnt == 1) ? C_buf[1] : 
    (s_cnt == 2) ? C_buf[2] : C_buf[3];

integer i, j;
always @(posedge clk or negedge rst_n) begin
    s_cur <= s_nxt;
    if (!rst_n) begin
        K_reg <= 0;
        M_reg <= 0;
        N_reg <= 0;
        A_index_base <= 0;
        B_index_base <= 0;
        C_index_base <= 0;
        C_shift_1 <= 0;
        C_shift_2 <= 0;
        index_r <= 0;
        index_c <= 0;
        busy <= 0;
        s_cur <= IDLE;
        s_cnt <= 0;
        for (i = 0; i < 4; i = i + 1) begin
            A_buf[i] <= 0;
            B_buf[i] <= 0;
            C_buf[i] <= 0;
        end
    end
    else if (in_valid && !busy) begin
        K_reg <= K;
        M_reg <= M;
        N_reg <= N;
        busy <= 1;
    end
    else if (busy) begin
        case (s_cur)
            IDLE: begin
                // do nothing
            end
            SA_CLEAR_1: begin
                sa_clear <= 1;
            end
            SA_CLEAR_2: begin
                sa_clear <= 0;
            end
            READ_ID: begin
            end
            READ_DATA: begin
                A_buf[s_cnt] <= (A_index_base + s_cnt < K_reg ? A_data_out : 0);
                B_buf[s_cnt] <= (B_index_base + s_cnt < K_reg ? B_data_out : 0);
                if (s_cnt == 3) begin
                    s_cnt <= 0;
                    A_index_base <= A_index_base + 4;
                    B_index_base <= B_index_base + 4;
                end
                else
                    s_cnt <= s_cnt + 1;
            end
            SA_START: begin
                sa_in_valid <= 1;
            end
            SA_WAIT_1: begin
                sa_in_valid <= 0;
            end
            SA_WAIT_2: begin
                if (!sa_busy) begin
                    {C_buf[0], C_buf[1], C_buf[2], C_buf[3]} <= C_output;
                end
            end
            C_SAVE: begin
                if (s_cnt == 3) begin
                    s_cnt <= 0;
                    C_index_base <= C_index_base + 4;
                end
                else 
                    s_cnt <= s_cnt + 1;
            end
            NEXT_LOOP: begin
                A_index_base <= 0;
                B_index_base <= 0;
                C_index_base <= 0;
                if ((index_r + 1) * 4 >= M_reg) begin
                    index_r <= 0;
                    index_c <= index_c + 1;
                end
                else
                    index_r <= index_r + 1;
                if (C_shift_1 + 4 >= M_reg) begin
                    C_shift_2 <= C_shift_2 + M_reg;
                    C_shift_1 <= 0;
                end
                else
                    C_shift_1 <= C_shift_1 + 4;
            end
            RESET: begin
                K_reg <= 0;
                M_reg <= 0;
                N_reg <= 0;
                A_index_base <= 0;
                B_index_base <= 0;
                C_index_base <= 0;
                C_shift_1 <= 0;
                C_shift_2 <= 0;
                index_r <= 0;
                index_c <= 0;
                busy <= 0;
                s_cnt <= 0;
                for (i = 0; i < 4; i = i + 1) begin
                    A_buf[i] <= 0;
                    B_buf[i] <= 0;
                end
                busy <= 0;
            end
            default: begin
                // do nothing
            end
        endcase
    end
end

always @(*) begin
    case (s_cur)
        IDLE: s_nxt = (in_valid ? SA_CLEAR_1 : IDLE);
        SA_CLEAR_1: s_nxt = SA_CLEAR_2;
        SA_CLEAR_2: s_nxt = (sa_busy ? SA_CLEAR_2 : READ_ID);
        READ_ID: s_nxt = READ_DATA;
        READ_DATA: s_nxt = ((s_cnt == 3) ? SA_START : READ_ID);
        SA_START: s_nxt = SA_WAIT_1;
        SA_WAIT_1: s_nxt = SA_WAIT_2;
        SA_WAIT_2: s_nxt = (sa_busy ? SA_WAIT_2 : (A_index_base < K_reg ? READ_ID : C_SAVE));
        C_SAVE: s_nxt = ((s_cnt == 3) ? NEXT_LOOP : C_SAVE);
        NEXT_LOOP: s_nxt = (index_c * 4 < N_reg ? SA_CLEAR_1 : RESET);
        RESET: s_nxt = IDLE;
    endcase
end

endmodule