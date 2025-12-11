# AMD Hardware Design Verification Engineering Internship Study Plan
**Target Position:** Hardware Design Verification Engineering Intern/Co-op  
**Company:** AMD (Markham, Ontario, Canada)  
**Timeline:** October 2025 - May 2026 (8 months)  
**Application Target:** May 2026 Summer Internship

## Overview
This plan builds foundational knowledge first, then progresses to specialized verification and hardware design skills. Each month focuses on 2-3 core areas with practical projects.

---

## Month 1: October 2025 - Programming Foundations & Linux
**Focus:** Core programming skills and development environment setup

### Week 1-2: Python Mastery
- **Resources:** "Automate the Boring Stuff with Python" + LeetCode
- **Goals:**
  - Master Python syntax, data structures, OOP
  - File I/O, regular expressions, error handling
  - Practice 15-20 LeetCode problems
- **Project:** Build a log file analyzer script

### Week 3-4: Linux/Unix Environment
- **Resources:** "The Linux Command Line" book + hands-on practice
- **Goals:**
  - Command line proficiency (bash, shell scripting)
  - File permissions, process management
  - Text processing (grep, sed, awk)
  - Basic networking commands
- **Project:** Create shell scripts for file management and system monitoring

### Monthly Milestone:
- Complete a Python project that processes hardware simulation logs
- Write shell scripts to automate development workflow

---

## Month 2: November 2025 - C/C++ & Digital Systems
**Focus:** Systems programming and digital logic fundamentals

### Week 1-2: C Programming
- **Resources:** "C Programming: A Modern Approach" + practice problems
- **Goals:**
  - Pointers, memory management, data structures
  - File handling, bit manipulation
  - Understanding hardware-software interface
- **Project:** Implement basic data structures in C

### Week 3-4: Digital Logic Fundamentals
- **Resources:** "Digital Design and Computer Architecture" (Harris & Harris)
- **Goals:**
  - Boolean algebra, logic gates, truth tables
  - Combinational and sequential circuits
  - Finite State Machines (FSM)
  - Timing analysis, setup/hold times
- **Project:** Design FSMs for simple control systems

### Monthly Milestone:
- Build a C program that simulates basic logic gates
- Design and document an FSM for a traffic light controller

---

## Month 3: December 2025 - Computer Architecture & Hardware
**Focus:** CPU design and computer organization

### Week 1-2: Computer Organization
- **Resources:** "Computer Organization and Design" (Patterson & Hennessy)
- **Goals:**
  - CPU architecture (datapath, control unit)
  - Instruction sets, assembly language
  - Memory hierarchy, caching
  - Pipeline concepts
- **Project:** Simulate a simple CPU instruction execution

### Week 3-4: Hands-on Hardware
- **Goals:**
  - Build a PC from components
  - Install multiple operating systems
  - Configure BIOS/UEFI, drivers
  - Basic network configuration
  - Graphics card installation and setup
- **Project:** Document complete PC build process with troubleshooting guide

### Monthly Milestone:
- Complete PC build with Linux/Windows dual boot
- Understand CPU instruction pipeline through simulation

---

## Month 4: January 2026 - Verilog & HDL Basics
**Focus:** Hardware description languages

### Week 1-2: Verilog Fundamentals
- **Resources:** "Verilog HDL: A Guide to Digital Design and Synthesis" (Palnitkar)
- **Goals:**
  - Verilog syntax, modules, ports
  - Behavioral vs structural modeling
  - Testbenches and simulation
  - Synthesis concepts
- **Project:** Implement basic combinational circuits in Verilog

### Week 3-4: Sequential Logic in Verilog
- **Goals:**
  - Flip-flops, latches, registers
  - Finite state machines in Verilog
  - Clock domains and timing
  - Simple arithmetic units
- **Project:** Design a counter-based system with FSM control

### Monthly Milestone:
- Create Verilog modules for ALU components
- Write comprehensive testbenches with waveform analysis

---

## Month 5: February 2026 - SystemVerilog & Advanced HDL
**Focus:** Advanced verification concepts

### Week 1-2: SystemVerilog Features
- **Resources:** "SystemVerilog for Verification" (Spear & Tumbush)
- **Goals:**
  - SystemVerilog enhancements over Verilog
  - Interfaces, packages, classes
  - Constrained random testing
  - Assertions (SVA basics)
- **Project:** Convert Verilog designs to SystemVerilog with interfaces

### Week 3-4: Verification Methodology
- **Goals:**
  - Testbench architecture
  - Coverage-driven verification
  - Functional coverage concepts
  - Basic UVM components understanding
- **Project:** Build a comprehensive testbench for previous month's design

### Monthly Milestone:
- Implement SystemVerilog testbench with constrained random testing
- Understand verification planning and coverage metrics

---

## Month 6: March 2026 - C++ & Advanced Programming
**Focus:** Object-oriented programming and scripting

### Week 1-2: C++ Programming
- **Resources:** "Effective C++" (Scott Meyers) + practice projects
- **Goals:**
  - OOP concepts, STL, templates
  - Memory management, smart pointers
  - Design patterns relevant to hardware tools
- **Project:** Build a log parser with configurable analysis rules

### Week 3-4: Perl & Database Basics
- **Goals:**
  - Perl syntax for text processing and automation
  - Regular expressions mastery
  - MySQL basics for test tracking
  - PHP fundamentals for web interfaces
- **Project:** Create regression tracking system using Perl/MySQL

### Monthly Milestone:
- Develop C++ application for hardware data analysis
- Build Perl scripts for test automation and reporting

---

## Month 7: April 2026 - UVM & Advanced Verification
**Focus:** Industry-standard verification methodology

### Week 1-2: UVM Fundamentals
- **Resources:** "UVM Primer" + online tutorials
- **Goals:**
  - UVM testbench structure
  - Agents, drivers, monitors, scoreboard
  - Transaction-level modeling
  - Configuration and factory patterns
- **Project:** Build simple UVM testbench

### Week 3-4: Debugging & Tools
- **Goals:**
  - Simulation debugging techniques
  - Waveform analysis
  - Protocol checkers
  - Performance optimization
- **Project:** Debug complex verification scenarios

### Monthly Milestone:
- Complete UVM testbench for a complex design
- Demonstrate debugging skills with simulation tools

---

## Month 8: May 2026 - Integration & Portfolio
**Focus:** Putting it all together and application preparation

### Week 1-2: System-Level Integration
- **Goals:**
  - Hardware-software co-verification
  - System-level testing concepts
  - Performance validation
  - Production testing basics
- **Project:** End-to-end verification of a complete system

### Week 3-4: Portfolio & Interview Prep
- **Goals:**
  - Document all projects in a portfolio
  - Prepare for technical interviews
  - Review AMD-specific technologies (RDNA, Zen architecture)
  - Practice explaining complex technical concepts
- **Activities:**
  - GitHub portfolio organization
  - Mock technical interviews
  - AMD technology research

### Monthly Milestone:
- Complete technical portfolio showcasing verification skills
- Ready for AMD internship application and interviews

---

## Key Tools & Software to Master
- **Simulation Tools:** ModelSim, QuestaSim, or open-source alternatives (Icarus Verilog, Verilator)
- **Development:** Git, make, debugging tools (gdb, valgrind)
- **IDEs:** VS Code with relevant extensions, vim/emacs
- **Operating Systems:** Ubuntu/CentOS Linux, Windows
- **Version Control:** Git with branching strategies

## Recommended Study Schedule
- **Daily:** 2-3 hours on weekdays, 4-5 hours on weekends
- **Weekly:** Complete one major project or milestone
- **Practice:** Daily coding in relevant languages
- **Review:** Weekly review of previous month's concepts

## Portfolio Projects to Highlight
1. **CPU Design Verification:** Complete verification environment for a simple processor
2. **SystemVerilog Testbench:** Advanced testbench with constrained random testing
3. **Automation Scripts:** Perl/Python tools for regression management
4. **Hardware Project:** PC build documentation with troubleshooting
5. **UVM Project:** Production-quality verification environment

## AMD-Specific Preparation
- Study AMD GPU architectures (RDNA) and CPU designs (Zen)
- Understand AI/ML hardware acceleration concepts
- Research AMD's verification methodologies and tools
- Follow AMD technical blogs and publications

## Application Timeline
- **March 2026:** Begin monitoring AMD internship postings
- **April 2026:** Submit applications with completed portfolio
- **May 2026:** Interview preparation and start date coordination

---

**Success Metrics:**
- Fluency in Verilog/SystemVerilog for design and verification
- Proficiency in multiple programming languages (Python, C++, Perl)
- Understanding of digital design and computer architecture
- Hands-on experience with verification methodologies
- Strong portfolio demonstrating practical skills
- Readiness for technical interviews and real-world verification challenges
