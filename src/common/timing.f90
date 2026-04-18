module timing_module
    implicit none
    
    integer(kind=8) :: count_rate
    integer(kind=8) :: start_math, end_math
    integer(kind=8) :: start_io, end_io
    real(kind=8)    :: total_math_time = 0.0d0
    real(kind=8)    :: total_io_time = 0.0d0
    integer(kind=8) :: total_steps = 0

contains

    subroutine init_timer()
        call system_clock(count_rate=count_rate)
        total_math_time = 0.0d0
        total_io_time = 0.0d0
        total_steps = 0
        print *, "Hardware Timer Initialized. Clock Rate: ", count_rate
    end subroutine init_timer

    subroutine start_math_timer()
        call system_clock(count=start_math)
    end subroutine start_math_timer

    subroutine stop_math_timer()
        call system_clock(count=end_math)
        total_math_time = total_math_time + real(end_math - start_math, 8) / real(count_rate, 8)
        total_steps = total_steps + 1
    end subroutine stop_math_timer

    subroutine start_io_timer()
        call system_clock(count=start_io)
    end subroutine start_io_timer

    subroutine stop_io_timer()
        call system_clock(count=end_io)
        total_io_time = total_io_time + real(end_io - start_io, 8) / real(count_rate, 8)
    end subroutine stop_io_timer

    subroutine print_timing_results(nx, ny, arch)
        integer, intent(in) :: nx, ny
        character(len=*), intent(in) :: arch
        real(kind=8) :: mcups
        logical :: file_exists

        if (total_math_time > 0.0d0) then
            mcups = real(nx, 8) * real(ny, 8) * real(total_steps, 8) / (total_math_time * 1.0d6)
        else
            mcups = 0.0d0
        end if

        ! --- 1. Print to Terminal ---
        print *, "------------------------------------------------"
        print *, "               BENCHMARK RESULTS                "
        print *, "------------------------------------------------"
        print *, "Architecture:      ", trim(arch)
        print *, "Resolution:        ", nx, "x", ny
        print *, "Total Steps:       ", total_steps
        print *, "Math Time (Compute): ", total_math_time, " seconds"
        print *, "I/O Time (Disk/PCIe):", total_io_time, " seconds"
        print *, "Total Wall-Clock:  ", total_math_time + total_io_time, " seconds"
        print *, "Performance:       ", mcups, " MCUPS"
        print *, "------------------------------------------------"

        ! --- 2. Append to CSV ---
        inquire(file="benchmarks.csv", exist=file_exists)
        if (file_exists) then
            open(unit=20, file="benchmarks.csv", position="append", status="old", action="write")
        else
            open(unit=20, file="benchmarks.csv", status="new", action="write")
            write(20, '(A)') "Language,Architecture,GridSize,MathTime_sec,IOTime_sec,MCUPS"
        end if        
        write(20, '(A,A,A,A,I0,A,F0.6,A,F0.6,A,F0.6)') &
            "Fortran", ",", trim(arch), ",", nx, ",", total_math_time, ",", total_io_time, ",", mcups
        close(20)

    end subroutine print_timing_results

end module timing_module