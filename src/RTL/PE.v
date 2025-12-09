module PE(
    clk,
    rst_n,
    in_valid,
    clear,

    row_in,
    row_out,
    col_in,
    col_out,
    data_out
);

input clk;
input rst_n;
input in_valid;
input clear;

input signed [8:0] row_in;
input signed [7:0] col_in;
output reg signed [8:0] row_out;
output reg signed [7:0] col_out;
output signed [31:0] data_out;

reg signed [31:0] output_reg;

assign data_out = output_reg;

always @(posedge clk or negedge rst_n) begin
    if (~rst_n) begin
        row_out <= 0;
        col_out <= 0;
        output_reg <= 0;
    end
    else begin
        if (clear) begin
            output_reg <= 0;
        end
        else if (in_valid) begin
            row_out <= row_in;
            col_out <= col_in;
            output_reg <= output_reg + row_in * $signed({col_in[7], col_in});
        end
    end
end

endmodule