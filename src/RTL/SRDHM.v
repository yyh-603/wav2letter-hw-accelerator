module SRDHM(
    input signed [31:0] a,
    input signed [31:0] b,
    output signed [31:0] out_x
);

wire overflow;

wire signed [63:0] ab_64;
wire signed [31:0] nudge;
wire signed [63:0] ab_64_nudge;
wire signed [63:0] neg_ab_64_nudge;
wire signed [31:0] shifted_ab_64_mudge;
wire signed [31:0] shifted_neg_ab_64_mudge;

assign overflow        = (a == 32'h80000000) && (b == 32'h80000000);
assign ab_64           = $signed(a) * $signed(b);
assign nudge           = ab_64[63] ? 32'hc0000001 : 32'h40000000;
assign ab_64_nudge     = ab_64 + {{32{nudge[31]}}, nudge};
assign neg_ab_64_nudge = -ab_64_nudge;

assign shifted_ab_64_mudge     = ab_64_nudge >>> 31;
assign shifted_neg_ab_64_mudge = neg_ab_64_nudge >>> 31;

assign out_x = overflow ? 32'h7fffffff : 
               ab_64_nudge[63] ? -shifted_neg_ab_64_mudge : shifted_ab_64_mudge;

endmodule