module systolic_array(
    clk,
    rst_n,
    in_valid,
    clear,

    row_in,
    col_in,
    input_offset,
    data_out,
    busy
);

input clk;
input rst_n;
input in_valid;
input clear;

input [127:0] row_in;
input [127:0] col_in;
input signed [8:0] input_offset;
output [511:0] data_out;
output reg busy;

wire [8:0] row_connect [0:3][0:4];
wire [7:0] col_connect [0:4][0:3];

reg [8:0] row_in_reg [0:3][0:3];
reg [7:0] col_in_reg [0:3][0:3];

wire [7:0] row_in_wire [0:3][0:3];
wire [7:0] col_in_wire [0:3][0:3];

reg [3:0] busy_cnt;
reg signed [3:0] k_cnt;

wire pe_valid;

assign pe_valid = busy_cnt > 0;

genvar r, c;
for (r = 0; r < 4; r = r + 1) begin : ROW_INPUT
    localparam signed [3:0] R = r[3:0];
    wire signed [3:0] kmin_r = R;
    wire signed [3:0] kmax_r = R + 4'd4;
    assign row_connect[r][0] =
        ((k_cnt >= kmin_r) && (k_cnt < kmax_r)) ? $signed(row_in_reg[k_cnt - R][r]) + input_offset : 9'sd0;
end
for (c = 0; c < 4; c = c + 1) begin : COL_INPUT
    localparam [3:0] C = c[3:0];
    wire signed [3:0] kmin_c = C;
    wire signed [3:0] kmax_c = C + 4'd4;
    assign col_connect[0][c] =
        ((k_cnt >= kmin_c) && (k_cnt < kmax_c)) ? col_in_reg[k_cnt - C][c] : 8'sd0;
end

generate
    for (r = 0; r < 4; r = r + 1) begin : ROW 
        for (c = 0; c < 4; c = c + 1) begin : COL 
            PE u_pe(
                .clk(clk),
                .rst_n(rst_n),
                .in_valid(pe_valid),
                .clear(clear),

                .row_in(row_connect[r][c]),
                .row_out(row_connect[r][c + 1]),
                .col_in(col_connect[r][c]),
                .col_out(col_connect[r + 1][c]),
                .data_out(data_out[128 * (3 - r) + 32 * (3 - c) + 31: 128 * (3 - r) + 32 * (3 - c)])
            );
        end
    end
endgenerate

generate
    for (r = 0; r < 4; r = r + 1) begin
        for (c = 0; c < 4; c = c + 1) begin
            assign row_in_wire[3 - r][3 - c] = row_in[32 * r + 8 * c + 7: 32 * r + 8 * c];
            assign col_in_wire[3 - r][3 - c] = col_in[32 * r + 8 * c + 7: 32 * r + 8 * c];
        end
    end
endgenerate

integer i, j;
always@(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < 4; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                row_in_reg[i][j] <= 0;
                col_in_reg[i][j] <= 0;
            end
        end
        busy_cnt <= 0;
        k_cnt <= 0;
    end
    else begin
        if (clear) begin
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    row_in_reg[i][j] <= 0;
                    col_in_reg[i][j] <= 0;
                end
            end
            busy_cnt <= 4'd10;
            busy <= 1;
            k_cnt <= 0;
        end 
        else if (in_valid) begin
            for (i = 0; i < 4; i = i + 1) begin
                for (j = 0; j < 4; j = j + 1) begin
                    row_in_reg[i][j] <= {row_in_wire[i][j][7], row_in_wire[i][j]};
                    col_in_reg[i][j] <= {col_in_wire[i][j]};
                end
            end
            busy_cnt <= 4'd10;
            busy <= 1;
            k_cnt <= 0;
        end
        else if (busy_cnt != 0) begin
            busy_cnt <= busy_cnt - 1;
            k_cnt <= k_cnt + 1;
            busy <= 1;
        end
        else begin
            busy <= 0;
        end
    end
end

endmodule