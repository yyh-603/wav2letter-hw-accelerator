module RDBPOT(
    input signed [31:0] in_x,
    input signed [31:0] exp,
    output signed [31:0] out_x
);

wire signed [31:0] mask;
wire signed [31:0] remainder;
wire signed [31:0] threshold;
wire signed [31:0] shifted;

assign mask      = (32'd1 << exp) - 32'd1;
assign remainder = in_x & mask;
assign threshold = (mask >> 1) + in_x[31];

assign shifted   = in_x >>> exp;

assign out_x = shifted + (remainder > threshold ? 32'd1 : 32'd0);

endmodule