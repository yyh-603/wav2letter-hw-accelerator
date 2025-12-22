`include "./src/RTL/TPU.v"
`include "./src/RTL/global_buffer.v"
`include "./src/RTL/RDBPOT.v"
`include "./src/RTL/SRDHM.v"

// make ENABLE_TRACE_ARG=--trace VERILATOR_TRACE_DEPTH=2 renode

module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output              rsp_valid,
  input               rsp_ready,
  output     [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

// funct7 = 1: load A
// funct7 = 2: load B
// funct7 = 3: run tpu
// funct7 = 4: read C
// funct7 = 5: reset C

wire [2:0] funct3 = cmd_payload_function_id[2:0];
wire [6:0] funct7 = cmd_payload_function_id[9:3];

wire sel_a =     (funct7 == 7'd1) && (funct3 == 3'd0);
wire sel_b =     (funct7 == 7'd2) && (funct3 == 3'd0);
wire sel_run =   (funct7 == 7'd3) && (funct3 == 3'd0);
wire sel_c =     (funct7 == 7'd4) && (funct3 == 3'd0);
wire sel_rst_c = (funct7 == 7'd5) && (funct3 == 3'd0);

wire sel_rdbpot = (funct7 == 7'd1) && (funct3 == 3'd1);
wire sel_srdhm  = (funct7 == 7'd2) && (funct3 == 3'd1);

//------------------------------------------------------------
// CFU State Machine
//------------------------------------------------------------

localparam S_IDLE        = 3'd0;
localparam S_READ_WAIT   = 3'd1;
localparam S_READ_WAIT_2 = 3'd2;
localparam S_RUN         = 3'd3;
localparam S_RESP        = 3'd4;
localparam S_RST_C       = 3'd5;

reg [2:0]  state_q, state_d;

reg        rsp_valid_q;
reg [31:0] rsp_payload_q;

assign rsp_valid             = rsp_valid_q;
assign rsp_payload_outputs_0 = rsp_payload_q;

wire cmd_fire = cmd_valid && cmd_ready;
assign cmd_ready = (state_q == S_IDLE) && ~rsp_valid_q && ~tpu_busy;

always @(*) begin
    state_d = state_q;
    case (state_q)
        S_IDLE: begin
            if (cmd_fire && sel_c)
                state_d = S_READ_WAIT;
            else if (cmd_fire && (sel_a || sel_b))
                state_d = S_RESP;
            else if (cmd_fire && sel_run)
                state_d = S_RUN;
            else if (cmd_fire && sel_rst_c)
                state_d = S_RST_C;
        end
        S_READ_WAIT: begin
            state_d = S_READ_WAIT_2;
        end
        S_READ_WAIT_2: begin
            state_d = S_RESP;
        end
        S_RUN: begin
            if (~tpu_busy)
                state_d = S_RESP;
        end
        S_RST_C: begin
            if (clear_done)
                state_d = S_RESP;
        end
        S_RESP: begin
            if (rsp_valid_q && rsp_ready)
                state_d = S_IDLE;
        end 
        default: state_d = S_IDLE;
    endcase
end

always @(posedge clk or posedge reset) begin
    if (reset) begin
        state_q <= S_IDLE;
    end
    else begin 
        state_q <= state_d;
    end;
end

always @(posedge clk or posedge reset) begin
    if (reset) begin
        input_K_reg      <= 9'd0;
        input_M_reg      <= 9'd0;
        input_N_reg      <= 9'd0;
        input_offset_reg <= 9'd0;
    end else if (cmd_fire && sel_run) begin
        input_K_reg      <= cmd_payload_inputs_0[26:18];
        input_M_reg      <= cmd_payload_inputs_0[17:9];
        input_N_reg      <= cmd_payload_inputs_0[8:0];
        input_offset_reg <= cmd_payload_inputs_1[8:0];
    end
end

always @(posedge clk or posedge reset) begin
    if (reset) begin
        rsp_valid_q   <= 1'b0;
        rsp_payload_q <= 32'd0;
    end
    else begin
        if (rsp_valid_q && !rsp_ready) begin
            rsp_valid_q <= 1'b1;
            rsp_payload_q <= rsp_payload_q;
        end
        else begin
            case (state_q)
                S_IDLE: begin
                    if (cmd_fire && (sel_a || sel_b)) begin
                        rsp_valid_q   <= 1'b1;
                        rsp_payload_q <= 32'd0;
                    end
                    else if (cmd_fire && sel_rdbpot) begin
                        rsp_valid_q   <= 1'b1;
                        rsp_payload_q <= rdbpot_out;
                    end
                    else if (cmd_fire && sel_srdhm) begin
                        rsp_valid_q   <= 1'b1;
                        rsp_payload_q <= srdhm_out;
                    end
                    else begin
                        rsp_valid_q   <= 1'b0;
                        rsp_payload_q <= 32'd0;
                    end
                end
                
                S_READ_WAIT: begin

                end

                S_READ_WAIT_2: begin
                    rsp_valid_q   <= 1'b1;
                    rsp_payload_q <= C_word;
                end

                S_RUN: begin
                    if (~tpu_busy) begin
                        rsp_valid_q <= 1'd1;
                        rsp_payload_q <= 32'd0;
                    end
                    else begin
                        rsp_valid_q <= 1'b0;
                        rsp_payload_q <= 32'd0;
                    end
                end

                S_RESP: begin
                    if (rsp_ready) begin
                        rsp_valid_q   <= 1'b0;
                        rsp_payload_q <= 32'd0;
                    end
                end

                S_RST_C: begin
                    if (clear_done) begin
                        rsp_valid_q <= 1'd1;
                        rsp_payload_q <= 32'd0;
                    end
                    else begin
                        rsp_valid_q <= 1'd0;
                        rsp_payload_q <= 32'd0;
                    end
                end

                default: begin
                    rsp_valid_q   <= 1'b0;
                    rsp_payload_q <= 32'd0;
                end
            endcase
        end
    end
end


//------------------------------------------------------------
// A Buffer
//------------------------------------------------------------

wire [13:0] A_addr;
wire [31:0] A_din;
wire [31:0] A_dout;
wire        A_we;

wire cpu_A_we = cmd_fire && sel_a;

global_buffer_bram #(
    .ADDR_BITS(14),
    .DATA_BITS(32)
) buffer_A(
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(A_we),
    .index(A_addr),
    .data_in(A_din),
    .data_out(A_dout)
);

assign A_we   = cpu_A_we ? 1'b1 : tpu_A_we;
assign A_addr = cpu_A_we ? cmd_payload_inputs_0[13:0] : tpu_A_addr;
assign A_din  = cpu_A_we ? cmd_payload_inputs_1 : tpu_A_din;

//------------------------------------------------------------
// B Buffer
//------------------------------------------------------------

wire [12:0] B_addr;
wire [31:0] B_din;
wire [31:0] B_dout;
wire        B_we;

wire cpu_B_we = cmd_fire && sel_b;

global_buffer_bram #(
    .ADDR_BITS(13),
    .DATA_BITS(32)
) buffer_B(
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(B_we),
    .index(B_addr),
    .data_in(B_din),
    .data_out(B_dout)
);

assign B_we =   cpu_B_we ? 1'b1 : tpu_B_we;
assign B_addr = cpu_B_we ? cmd_payload_inputs_0[12:0] : tpu_B_addr;
assign B_din =  cpu_B_we ? cmd_payload_inputs_1 : tpu_B_din;


//------------------------------------------------------------
// C Buffer
//------------------------------------------------------------

wire [12:0]  C_addr;
wire [1:0]   C_addr_2;
wire [127:0] C_din;
wire [127:0] C_dout;
wire         C_we;

reg [12:0] C_addr_reg;
reg [1:0]  C_addr_2_reg;

reg [127:0] C_read_data_q;

wire [31:0] C_word = (C_addr_2_reg == 2'd0) ? C_dout[127:96] :
                     (C_addr_2_reg == 2'd1) ? C_dout[95:64]  :
                     (C_addr_2_reg == 2'd2) ? C_dout[63:32]  :
                                              C_dout[31:0];

localparam integer C_ADDR_BITS = 13;

wire C_acc_mode;

global_buffer_bram_acc #(
    .ADDR_BITS(C_ADDR_BITS),
    .DATA_BITS(128)
) buffer_C(
    .clk(clk),
    .rst_n(~reset),
    .ram_en(1'b1),
    .wr_en(C_we),
    .acc_mode(C_acc_mode),
    .index(C_addr),
    .data_in(C_din),
    .data_out(C_dout)
);

// global_buffer_bram #(
//     .ADDR_BITS(12),
//     .DATA_BITS(128)
// ) buffer_C(
//     .clk(clk),
//     .rst_n(~reset),
//     .ram_en(1'b1),
//     .wr_en(C_we),
//     .index(C_addr),
//     .data_in(C_din),
//     .data_out(C_dout)
// );

assign C_we       = c_clear_one_q ? 1'b1 : (cpu_C_read_active ? 1'b0 : tpu_C_we);
assign C_addr     = c_clear_one_q ? c_clear_addr_q : (cpu_C_read_active ? C_addr_reg : tpu_C_addr);
assign C_din      = c_clear_one_q ? 128'd0 : tpu_C_din;
assign C_acc_mode = c_clear_one_q ? 1'b0 : (cpu_C_read_active ? 1'b0 : tpu_C_we);

always @(posedge clk or posedge reset) begin
    if (reset) begin
        C_addr_reg   <= 13'd0;
        C_addr_2_reg <= 2'd0;
        C_read_data_q <= 128'd0;
    end else begin
        if (cmd_fire && sel_c && state_q == S_IDLE) begin
            C_addr_reg   <= cmd_payload_inputs_0[12:0];
            C_addr_2_reg <= cmd_payload_inputs_1[1:0];
        end

        if (state_q == S_READ_WAIT) begin
            C_read_data_q <= C_dout;
        end
    end
end

reg c_read_active_q;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        c_read_active_q <= 1'b0;
    end else begin
        if (cmd_fire && sel_c && state_q == S_IDLE) begin
            c_read_active_q <= 1'b1;
        end
        else if (c_read_active_q && state_q == S_RESP && rsp_valid_q && rsp_ready) begin
            c_read_active_q <= 1'b0;
        end
    end
end

wire cpu_C_read_active = c_read_active_q;


reg                   c_clear_one_q;
reg [C_ADDR_BITS-1:0] c_clear_addr_q;

wire clear_done = (state_q == S_RST_C) && c_clear_one_q;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        c_clear_one_q  <= 1'b0;
        c_clear_addr_q <= {C_ADDR_BITS{1'b0}};
    end else begin
        if (cmd_fire && sel_rst_c && (state_q == S_IDLE)) begin
            c_clear_one_q  <= 1'b1;
            c_clear_addr_q <= cmd_payload_inputs_0[12:0];
        end
        else if (state_q == S_RST_C) begin
            c_clear_one_q <= 1'b0;
            c_clear_addr_q <= {C_ADDR_BITS{1'b0}};
        end
    end
end

//------------------------------------------------------------
// TPU
//------------------------------------------------------------

reg tpu_in_valid_q;
always @(posedge clk or posedge reset) begin
    if (reset) begin
        tpu_in_valid_q <= 1'b0;
    end else begin
        tpu_in_valid_q <= cmd_fire && sel_run;
    end
end
wire tpu_in_valid = tpu_in_valid_q;

reg [8:0] input_K_reg;
reg [8:0] input_M_reg;
reg [8:0] input_N_reg;
reg [8:0] input_offset_reg;

wire tpu_busy;

wire tpu_A_we;
wire tpu_B_we;
wire tpu_C_we;

wire [13:0] tpu_A_addr;
wire [12:0] tpu_B_addr;
wire [12:0] tpu_C_addr;

wire [31:0] tpu_A_din;
wire [31:0] tpu_B_din;
wire [127:0] tpu_C_din;

wire [31:0] tpu_A_data_out;
wire [31:0] tpu_B_data_out;
wire [127:0] tpu_C_data_out;

TPU tpu (
    .clk(clk),
    .rst_n(~reset),
    .in_valid(tpu_in_valid),
    .K(input_K_reg),
    .M(input_M_reg),
    .N(input_N_reg),
    .input_offset(input_offset_reg),
    .busy(tpu_busy),
    
    .A_wr_en(tpu_A_we),
    .A_index(tpu_A_addr),
    .A_data_in(tpu_A_din),
    .A_data_out(A_dout),

    .B_wr_en(tpu_B_we),
    .B_index(tpu_B_addr),
    .B_data_in(tpu_B_din),
    .B_data_out(B_dout),

    .C_wr_en(tpu_C_we),
    .C_index(tpu_C_addr),
    .C_data_in(tpu_C_din),
    .C_data_out(C_dout)
);

wire signed [31:0] rdbpot_out;
RDBPOT m_rdbpot (
    .in_x(cmd_payload_inputs_0),
    .exp(cmd_payload_inputs_1[4:0]),
    .out_x(rdbpot_out)
);

wire signed [31:0] srdhm_out;
SRDHM m_srdhm (
    .a(cmd_payload_inputs_0),
    .b(cmd_payload_inputs_1),
    .out_x(srdhm_out)
);


endmodule
