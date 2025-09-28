```markdown
# Verilog: Why `assign` works but `<=` doesn't for wire implementation

## Question
```verilog
module top_module( input in, output out );
    assign out = in;  // ✅ This works
    // out <= in;     // ❌ This doesn't work
endmodule
```

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
