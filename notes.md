## Assign VS <=
Why assign out = in; works:

Continuous Assignment: assign creates a continuous assignment that works outside of always blocks
Wire Type: Your output out is implicitly declared as a wire type
Always Active: The assignment is always "listening" - whenever in changes, out immediately follows

Why out <= in; doesn't work:

Procedural Assignment: <= is a non-blocking procedural assignment
Requires Always Block: Procedural assignments must be inside always blocks
Wrong Data Type: Procedural assignments typically drive reg types, not wire types
Missing Context: The synthesizer doesn't know when to execute this assignment

To make <= work, you'd need:
verilogmodule top_module( 
    input in, 
    output reg out  // Note: changed to 'reg'
);
    always @(*) begin    // Always block required
        out <= in;       // Now this works
    end
endmodule
Or with blocking assignment:
verilogmodule top_module( 
    input in, 
    output reg out 
);
    always @(*) begin
        out = in;        // Blocking assignment
    end
endmodule
Key Rules:

Wires: Can only be driven by assign statements or module outputs
Regs: Can only be driven by procedural assignments inside always blocks
Continuous vs Procedural: Different assignment mechanisms for different modeling styles
