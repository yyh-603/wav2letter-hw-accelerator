module global_buffer_bram #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
  input                      clk,
  input                      rst_n,
  input                      ram_en,
  input                      wr_en,
  input      [ADDR_BITS-1:0] index,
  input      [DATA_BITS-1:0] data_in,
  output reg [DATA_BITS-1:0] data_out
  );

  parameter DEPTH = 2**ADDR_BITS;

  (* ram_style = "block" *)
  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (posedge clk) begin
    if (ram_en) begin
      if(wr_en) begin
        gbuff[index] <= data_in;
      end
      else begin
        data_out <= gbuff[index];
      end
    end
  end

endmodule

module global_buffer_bram_acc #(parameter ADDR_BITS=8, parameter DATA_BITS=128)(
  input                      clk,
  input                      rst_n,
  input                      ram_en,
  input                      wr_en,
  input                      acc_mode,
  input      [ADDR_BITS-1:0] index,
  input      [DATA_BITS-1:0] data_in,
  output reg signed [DATA_BITS-1:0] data_out
  );

  parameter DEPTH = 2**ADDR_BITS;

  (* ram_style = "block" *)
  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (posedge clk) begin
    if (ram_en) begin
      if (wr_en) begin
        if (acc_mode) begin
          gbuff[index][31:0]   <= $signed(gbuff[index][31:0])   + $signed(data_in[31:0]);
          gbuff[index][63:32]  <= $signed(gbuff[index][63:32])  + $signed(data_in[63:32]);
          gbuff[index][95:64]  <= $signed(gbuff[index][95:64])  + $signed(data_in[95:64]);
          gbuff[index][127:96] <= $signed(gbuff[index][127:96]) + $signed(data_in[127:96]);
        end
        else begin
          gbuff[index] <= data_in;
        end
      end
      else begin
        data_out <= gbuff[index];
      end
    end
  end

endmodule
/**
  Example of instantiating a global_buffer_bram: 

  global_buffer_bram #(
    .ADDR_BITS(12), // ADDR_BITS 12 -> generates 2^12 entries
    .DATA_BITS(32)  // DATA_BITS 32 -> 32 bits for each entries
  )
  gbuff_A(
    .clk(clk),
    .rst_n(1'b1),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out)
  );

*/