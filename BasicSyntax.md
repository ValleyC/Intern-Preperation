Here's an overview of the basic Verilog syntax:

## Module Declaration
```verilog
module module_name (port_list);
    // module contents
endmodule
```

## Port Declarations
```verilog
module example (
    input wire clk,           // Clock input
    input wire [7:0] data_in, // 8-bit input
    output reg [7:0] data_out // 8-bit output
);
```

## Data Types

**Wire** - represents connections between gates
```verilog
wire signal;
wire [7:0] bus;  // 8-bit wire
```

**Reg** - represents storage elements
```verilog
reg flag;
reg [15:0] counter;  // 16-bit register
```

**Integer and Real**
```verilog
integer count;
real voltage;
```

## Number Formats
```verilog
8'b10101010    // 8-bit binary
4'h F          // 4-bit hexadecimal
32'd100        // 32-bit decimal
```

## Operators

**Arithmetic:** `+`, `-`, `*`, `/`, `%`
**Logical:** `&&`, `||`, `!`
**Bitwise:** `&`, `|`, `^`, `~`
**Relational:** `==`, `!=`, `<`, `>`, `<=`, `>=`
**Shift:** `<<`, `>>`

## Assignments

**Continuous Assignment**
```verilog
assign output_signal = input1 & input2;
```

**Procedural Assignment**
```verilog
always @(posedge clk) begin
    counter <= counter + 1;  // Non-blocking
    flag = 1'b1;             // Blocking
end
```

## Control Structures

**If-Else**
```verilog
if (condition)
    statement1;
else
    statement2;
```

**Case Statement**
```verilog
case (selector)
    2'b00: output = input0;
    2'b01: output = input1;
    default: output = 1'b0;
endcase
```

**For Loop**
```verilog
for (i = 0; i < 8; i = i + 1) begin
    array[i] = 1'b0;
end
```

## Always Blocks

**Combinational Logic**
```verilog
always @(*) begin
    // combinational logic
end
```

**Sequential Logic**
```verilog
always @(posedge clk or negedge reset) begin
    if (!reset)
        q <= 1'b0;
    else
        q <= d;
end
```

## Comments
```verilog
// Single line comment
/* Multi-line
   comment */
```

## Simple Example
```verilog
module and_gate (
    input wire a,
    input wire b,
    output wire y
);
    assign y = a & b;
endmodule
```

The key principles are that Verilog describes hardware behavior rather than software execution, with concurrent operations and explicit timing considerations.
