# Verilog: Why `assign` works but `<=` doesn't for wire implementation

## Question
```verilog
module top_module( input in, output out );
    assign out = in;  // ✅ This works
    // out <= in;     // ❌ This doesn't work
endmodule

## Why `assign out = in;` works:

1. **Continuous Assignment**: `assign` creates a continuous assignment that works outside of always blocks
2. **Wire Type**: Your output `out` is implicitly declared as a `wire` type
3. **Always Active**: The assignment is always "listening" - whenever `in` changes, `out` immediately follows

## Why `out <= in;` doesn't work:

1. **Procedural Assignment**: `<=` is a non-blocking procedural assignment
2. **Requires Always Block**: Procedural assignments must be inside `always` blocks
3. **Wrong Data Type**: Procedural assignments typically drive `reg` types, not `wire` types
4. **Missing Context**: The synthesizer doesn't know when to execute this assignment

## To make `<=` work, you'd need:

```verilog
module top_module( 
    input in, 
    output reg out  // Note: changed to 'reg'
);
    always @(*) begin    // Always block required
        out <= in;       // Now this works
    end
endmodule
```

Or with blocking assignment:
```verilog
module top_module( 
    input in, 
    output reg out 
);
    always @(*) begin
        out = in;        // Blocking assignment
    end
endmodule
```

## Key Rules:
- **Wires**: Can only be driven by `assign` statements or module outputs
- **Regs**: Can only be driven by procedural assignments inside `always` blocks
- **Continuous vs Procedural**: Different assignment mechanisms for different modeling styles

## Best Practice:
For simple wire connections like yours, `assign` is the preferred and most efficient approach.
```

# Verilog: Packed vs. Unpacked Arrays

## Quick Summary
- **Packed `[7:0]`**: Dimensions **BEFORE** name → bits packed together (vector)
- **Unpacked `[255:0]`**: Dimensions **AFTER** name → separate array elements (memory)

## Packed Arrays (Dimensions BEFORE name)
```verilog
reg [7:0] data;        // 8-bit vector (bits packed together)
wire [15:0] address;   // 16-bit vector


**Think of it as**: A single multi-bit value
- Can access individual bits: `data[3]`, `data[7:4]`
- Can use in arithmetic: `data + 1`
- Bits are stored contiguously

## Unpacked Arrays (Dimensions AFTER name)
```verilog
reg memory [255:0];    // 256 separate 1-bit elements
reg [7:0] ram [31:0];  // 32 separate 8-bit elements
```

**Think of it as**: An array of values (like C arrays)
- Access individual elements: `memory[0]`, `ram[15]`
- Each element is separate in memory
- Like declaring: `int array[256]` in C

## The Examples Explained

### Example 1: Memory Array
```verilog
reg [7:0] mem [255:0];
//  ^^^^^     ^^^^^^^^
//  PACKED    UNPACKED
//  (width)   (depth)
```

**Translation**: 
- An array of **256 elements** (unpacked dimension)
- Each element is **8 bits wide** (packed dimension)
- Like having 256 separate bytes

**Visual representation**:
```
mem[0]   = [7:0] = 8 bits
mem[1]   = [7:0] = 8 bits
mem[2]   = [7:0] = 8 bits
...
mem[255] = [7:0] = 8 bits
```

**How to access**:
```verilog
mem[0] = 8'hAB;        // Write to element 0
mem[10] = 8'hCD;       // Write to element 10
data = mem[5];         // Read element 5 (gets all 8 bits)
bit_value = mem[5][3]; // Read bit 3 of element 5
```

### Example 2: Simple Array
```verilog
reg mem2 [28:0];
//       ^^^^^^^
//       UNPACKED
//       (29 elements)
```

**Translation**:
- An array of **29 elements**
- Each element is **1 bit** (no packed dimension means 1-bit)

**Visual representation**:
```
mem2[0]  = 1 bit
mem2[1]  = 1 bit
mem2[2]  = 1 bit
...
mem2[28] = 1 bit
```

## Complete Examples

### Example 1: Register File (8 registers, each 32-bit)
```verilog
reg [31:0] register_file [7:0];
//  ^^^^^^                ^^^^^ 
//  32 bits wide          8 registers

// Usage:
register_file[0] = 32'h12345678;     // Write to register 0
register_file[5] = 32'hABCDEF00;     // Write to register 5
data = register_file[3];             // Read register 3
byte_val = register_file[2][7:0];    // Get lower byte of register 2
```

### Example 2: Simple RAM (256 bytes)
```verilog
reg [7:0] ram [255:0];
//  ^^^^^     ^^^^^^^^
//  byte      256 locations

// Usage:
ram[0] = 8'h42;           // Write to address 0
ram[100] = 8'hFF;         // Write to address 100
value = ram[50];          // Read from address 50
```

### Example 3: 2D Array-like Structure
```verilog
reg [3:0] matrix [7:0][7:0];
//  ^^^^^        ^^^^^ ^^^^^
//  4-bit data   8x8   array

// Usage:
matrix[0][0] = 4'b1010;   // Top-left element
matrix[7][7] = 4'b0101;   // Bottom-right element
```

## Comparison Table

| Declaration | What it means | Total bits |
|-------------|---------------|------------|
| `reg [7:0] a` | Single 8-bit vector | 8 |
| `reg a [7:0]` | 8 separate 1-bit elements | 8 |
| `reg [7:0] b [15:0]` | 16 elements, each 8-bit | 128 |
| `reg [3:0][7:0] c` | Single 32-bit vector (8 bits × 4) | 32 |

## Memory Declaration Pattern
```verilog
reg [DATA_WIDTH-1:0] memory_name [DEPTH-1:0];
//  ^^^^^^^^^^^^^^^^^              ^^^^^^^^^^^
//  How wide is each word          How many words
//  (PACKED)                       (UNPACKED)
```

## Common Use Cases

### Packed (Vector)
- Data buses
- Multi-bit signals
- Arithmetic operations
```verilog
wire [31:0] instruction;  // 32-bit instruction
wire [7:0] data_bus;      // 8-bit data
```

### Unpacked (Array/Memory)
- RAM/ROM
- Register files
- Lookup tables
```verilog
reg [7:0] ram [1023:0];        // 1KB RAM
reg [15:0] rom [255:0];        // 256-word ROM
reg [31:0] registers [15:0];   // 16 registers
```

## Key Takeaway
- **BEFORE the name `[7:0]`** = Packed = width of each element
- **AFTER the name `[255:0]`** = Unpacked = number of elements
- **Both together `[7:0] mem [255:0]`** = Array of multi-bit values
```
