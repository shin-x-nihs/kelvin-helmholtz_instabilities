program KHI_GPU_Benchmark
    use timing_module
    implicit none
    
    ! --- Simulation Parameters ---
    integer, parameter :: nx = 512, ny = 512     ! Scaled down for benchmark
    integer, parameter :: ng = 3                 ! Ghost cells for WENO5
    real, parameter    :: Lx = 1.0, Ly = 1.0
    real, parameter    :: dx = Lx / real(nx), dy = Ly / real(ny)
    real, parameter    :: CFL_target = 0.4       
    real, parameter    :: t_max = 2.0
    real, parameter    :: gamma = 1.4
    real, parameter    :: output_interval = 0.05 
    
    ! --- State Variables ---
    real, dimension(4, 1-ng:nx+ng, 1-ng:ny+ng) :: q, q1, q2, rhs
    real, dimension(4, 0:nx+1, 1:ny) :: F_ext
    real, dimension(4, 1:nx, 0:ny+1) :: G_ext
    
    integer :: i, j, frame
    real    :: x, y, rho, u, v, P, E
    real    :: t, dt, t_next_output
    
    print *, "Initializing 512x512 GPU Benchmark (OpenACC)..."
    call init_timer()
    
    ! 1. Initialize Domain
    q = 0.0
    do j = 1, ny
        do i = 1, nx
            y = (real(j) - 0.5) * dy
            x = (real(i) - 0.5) * dx
            
            if (y > 0.25 .and. y < 0.75) then
                u = 0.5; rho = 2.0
            else
                u = -0.5; rho = 1.0
            end if
            
            v = 0.05 * sin(4.0 * 3.14159265 * x) * &
                (exp(-(y-0.25)**2 / 0.005) + exp(-(y-0.75)**2 / 0.005)) 
            
            P = 2.5
            q(1, i, j) = rho
            q(2, i, j) = rho * u
            q(3, i, j) = rho * v
            q(4, i, j) = P/(gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
        end do
    end do
    
    t = 0.0
    frame = 0
    t_next_output = 0.0
    
    ! 2. Move data to GPU
    !$acc data copy(q) create(q1, q2, rhs, F_ext, G_ext)
    
    ! 3. Main Time Loop
    do while (t < t_max)
        
        ! --- Output Stage (TIMED SEPARATELY) ---
        if (t >= t_next_output - 1.0e-8) then
            call start_io_timer()
            !$acc update host(q)
            print *, "Time: ", t, "/", t_max, " | Frame: ", frame
            call write_vtk_binary(frame, q, nx, ny, ng, dx, dy)
            call stop_io_timer()
            
            frame = frame + 1
            t_next_output = t_next_output + output_interval
        end if
        
        ! --- Mathematical Computation Stage (MATH TIMER ON) ---
        call start_math_timer()
        
        call compute_dt(q, nx, ny, ng, dx, dy, gamma, CFL_target, dt)
        if (t + dt > t_next_output) dt = t_next_output - t
        
        ! SSP-RK3 Stage 1 
        call apply_periodic_bcs(q, nx, ny, ng)
        call compute_rhs(q, rhs, F_ext, G_ext, nx, ny, ng, dx, dy, gamma)
        !$acc parallel loop collapse(2) present(q, q1, rhs)
        do j = 1-ng, ny+ng
            do i = 1-ng, nx+ng
                q1(:, i, j) = q(:, i, j) + dt * rhs(:, i, j)
            end do
        end do
        
        ! SSP-RK3 Stage 2 
        call apply_periodic_bcs(q1, nx, ny, ng)
        call compute_rhs(q1, rhs, F_ext, G_ext, nx, ny, ng, dx, dy, gamma)
        !$acc parallel loop collapse(2) present(q, q1, q2, rhs)
        do j = 1-ng, ny+ng
            do i = 1-ng, nx+ng
                q2(:, i, j) = 0.75 * q(:, i, j) + 0.25 * q1(:, i, j) + 0.25 * dt * rhs(:, i, j)
            end do
        end do
        
        ! SSP-RK3 Stage 3 
        call apply_periodic_bcs(q2, nx, ny, ng)
        call compute_rhs(q2, rhs, F_ext, G_ext, nx, ny, ng, dx, dy, gamma)
        !$acc parallel loop collapse(2) present(q, q2, rhs)
        do j = 1-ng, ny+ng
            do i = 1-ng, nx+ng
                q(:, i, j) = 1.0/3.0 * q(:, i, j) + 2.0/3.0 * q2(:, i, j) + 2.0/3.0 * dt * rhs(:, i, j)
            end do
        end do
        
        call stop_math_timer()
        ! --- Mathematical Computation Stage (MATH TIMER OFF) ---
        
        t = t + dt
    end do
    
    !$acc end data
    
    ! Print Final Benchmark Output
    call print_timing_results(nx, ny, "OpenACC")

contains

    ! --- Adaptive Time-Step Calculator ---
    subroutine compute_dt(state, nx, ny, ng, dx, dy, gamma, cfl, dt_out)
        real, dimension(4, 1-ng:nx+ng, 1-ng:ny+ng), intent(in) :: state
        integer, intent(in) :: nx, ny, ng
        real, intent(in) :: dx, dy, gamma, cfl
        real, intent(out) :: dt_out
        real :: max_speed, rho, u, v, P, c
        integer :: i, j
        max_speed = 1.0e-8
        !$acc parallel loop collapse(2) present(state) reduction(max:max_speed) private(rho, u, v, P, c)
        do j = 1, ny
            do i = 1, nx
                rho = state(1, i, j)
                u = state(2, i, j) / rho
                v = state(3, i, j) / rho
                P = (gamma - 1.0) * (state(4, i, j) - 0.5 * rho * (u**2 + v**2))
                c = sqrt(gamma * P / rho)
                max_speed = max(max_speed, abs(u) + c, abs(v) + c)
            end do
        end do
        dt_out = cfl * min(dx, dy) / max_speed
    end subroutine compute_dt

    ! Making the function callable in a kernel
    !$acc routine seq               
    real function weno5(v1, v2, v3, v4, v5)
        real, intent(in) :: v1, v2, v3, v4, v5
        real :: b0, b1, b2, w0, w1, w2, a0, a1, a2, sum_a
        real, parameter :: eps = 1.0e-6
        b0 = 13.0/12.0*(v1 - 2.0*v2 + v3)**2 + 0.25*(v1 - 4.0*v2 + 3.0*v3)**2
        b1 = 13.0/12.0*(v2 - 2.0*v3 + v4)**2 + 0.25*(v2 - v4)**2
        b2 = 13.0/12.0*(v3 - 2.0*v4 + v5)**2 + 0.25*(3.0*v3 - 4.0*v4 + v5)**2
        a0 = 0.1 / (eps + b0)**2; a1 = 0.6 / (eps + b1)**2; a2 = 0.3 / (eps + b2)**2
        sum_a = a0 + a1 + a2
        w0 = a0 / sum_a; w1 = a1 / sum_a; w2 = a2 / sum_a
        weno5 = w0*(1.0/3.0*v1 - 7.0/6.0*v2 + 11.0/6.0*v3) + &
                w1*(-1.0/6.0*v2 + 5.0/6.0*v3 + 1.0/3.0*v4) + &
                w2*(1.0/3.0*v3 + 5.0/6.0*v4 - 1.0/6.0*v5)
    end function weno5

    subroutine apply_periodic_bcs(state, nx, ny, ng)
        real, dimension(4, 1-ng:nx+ng, 1-ng:ny+ng), intent(inout) :: state
        integer, intent(in) :: nx, ny, ng
        integer :: k
        !$acc parallel loop present(state)
        do k = 1, ng
            state(:, 1-k, 1:ny)    = state(:, nx-k+1, 1:ny)
            state(:, nx+k, 1:ny)   = state(:, k, 1:ny)
            state(:, 1-ng:nx+ng, 1-k)  = state(:, 1-ng:nx+ng, ny-k+1)
            state(:, 1-ng:nx+ng, ny+k) = state(:, 1-ng:nx+ng, k)
        end do
    end subroutine apply_periodic_bcs

    subroutine compute_rhs(state, rhs, F_ext, G_ext, nx, ny, ng, dx, dy, gamma)
        real, dimension(4, 1-ng:nx+ng, 1-ng:ny+ng), intent(in)  :: state
        real, dimension(4, 1-ng:nx+ng, 1-ng:ny+ng), intent(out) :: rhs
        real, dimension(4, 0:nx+1, 1:ny), intent(inout) :: F_ext
        real, dimension(4, 1:nx, 0:ny+1), intent(inout) :: G_ext
        integer, intent(in) :: nx, ny, ng
        real, intent(in)    :: dx, dy, gamma
        
        real :: qL(4), qR(4), prim(4, -2:3)
        real :: rhoL, uL, vL, PL, EL, cL, rhoR, uR, vR, PR, ER, cR
        real :: SL, SR, SM, PM
        real :: fL_flux(4), fR_flux(4), U_star_L(4), U_star_R(4)
        integer :: i, j, k, d
        
        !$acc parallel loop collapse(2) present(state, F_ext) &
        !$acc& private(prim, qL, qR, rhoL, uL, vL, PL, EL, cL, rhoR, uR, vR, PR, ER, cR, &
        !$acc& SL, SR, SM, PM, U_star_L, U_star_R, fL_flux, fR_flux)
        do j = 1, ny
            do i = 0, nx
                do k = -2, 3
                    prim(1, k) = state(1, i+k, j)
                    prim(2, k) = state(2, i+k, j) / prim(1, k)
                    prim(3, k) = state(3, i+k, j) / prim(1, k)
                    prim(4, k) = (gamma - 1.0) * (state(4, i+k, j) - 0.5*prim(1, k)*(prim(2, k)**2 + prim(3, k)**2))
                end do
                do d = 1, 4
                    qL(d) = weno5(prim(d,-2), prim(d,-1), prim(d,0), prim(d,1), prim(d,2))
                    qR(d) = weno5(prim(d,3),  prim(d,2),  prim(d,1), prim(d,0), prim(d,-1))
                end do
                rhoL = qL(1); uL = qL(2); vL = qL(3); PL = qL(4)
                EL = PL/(gamma - 1.0) + 0.5 * rhoL * (uL**2 + vL**2)
                cL = sqrt(gamma * PL / rhoL)
                fL_flux = [rhoL*uL, rhoL*uL**2 + PL, rhoL*uL*vL, (EL + PL)*uL]
                rhoR = qR(1); uR = qR(2); vR = qR(3); PR = qR(4)
                ER = PR/(gamma - 1.0) + 0.5 * rhoR * (uR**2 + vR**2)
                cR = sqrt(gamma * PR / rhoR)
                fR_flux = [rhoR*uR, rhoR*uR**2 + PR, rhoR*uR*vR, (ER + PR)*uR]
                SL = min(uL - cL, uR - cR)
                SR = max(uL + cL, uR + cR)
                if (SL >= 0.0) then
                    F_ext(:, i, j) = fL_flux
                else if (SR <= 0.0) then
                    F_ext(:, i, j) = fR_flux
                else
                    SM = (PR - PL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)) / (rhoL*(SL - uL) - rhoR*(SR - uR))
                    if (SM >= 0.0) then
                        PM = rhoL*(uL - SL)*(uL - SM) + PL
                        U_star_L = [rhoL*(SL - uL)/(SL - SM), rhoL*(SL - uL)/(SL - SM)*SM, rhoL*(SL - uL)/(SL - SM)*vL, rhoL*(SL - uL)/(SL - SM)*(EL/rhoL + (SM - uL)*(SM + PL/(rhoL*(SL - uL))))]
                        F_ext(:, i, j) = fL_flux + SL * (U_star_L - [rhoL, rhoL*uL, rhoL*vL, EL])
                    else
                        PM = rhoR*(uR - SR)*(uR - SM) + PR
                        U_star_R = [rhoR*(SR - uR)/(SR - SM), rhoR*(SR - uR)/(SR - SM)*SM, rhoR*(SR - uR)/(SR - SM)*vR, rhoR*(SR - uR)/(SR - SM)*(ER/rhoR + (SM - uR)*(SM + PR/(rhoR*(SR - uR))))]
                        F_ext(:, i, j) = fR_flux + SR * (U_star_R - [rhoR, rhoR*uR, rhoR*vR, ER])
                    end if
                end if
            end do
        end do
        
        !$acc parallel loop collapse(2) present(state, G_ext) &
        !$acc& private(prim, qL, qR, rhoL, uL, vL, PL, EL, cL, rhoR, uR, vR, PR, ER, cR, &
        !$acc& SL, SR, SM, PM, U_star_L, U_star_R, fL_flux, fR_flux)
        do i = 1, nx
            do j = 0, ny
                do k = -2, 3
                    prim(1, k) = state(1, i, j+k)
                    prim(2, k) = state(2, i, j+k) / prim(1, k)
                    prim(3, k) = state(3, i, j+k) / prim(1, k)
                    prim(4, k) = (gamma - 1.0) * (state(4, i, j+k) - 0.5*prim(1, k)*(prim(2, k)**2 + prim(3, k)**2))
                end do
                do d = 1, 4
                    qL(d) = weno5(prim(d,-2), prim(d,-1), prim(d,0), prim(d,1), prim(d,2))
                    qR(d) = weno5(prim(d,3),  prim(d,2),  prim(d,1), prim(d,0), prim(d,-1))
                end do
                rhoL = qL(1); uL = qL(2); vL = qL(3); PL = qL(4)
                EL = PL/(gamma - 1.0) + 0.5 * rhoL * (uL**2 + vL**2)
                cL = sqrt(gamma * PL / rhoL)
                fL_flux = [rhoL*vL, rhoL*uL*vL, rhoL*vL**2 + PL, (EL + PL)*vL]
                rhoR = qR(1); uR = qR(2); vR = qR(3); PR = qR(4)
                ER = PR/(gamma - 1.0) + 0.5 * rhoR * (uR**2 + vR**2)
                cR = sqrt(gamma * PR / rhoR)
                fR_flux = [rhoR*vR, rhoR*uR*vR, rhoR*vR**2 + PR, (ER + PR)*vR]
                SL = min(vL - cL, vR - cR)
                SR = max(vL + cL, vR + cR)
                if (SL >= 0.0) then
                    G_ext(:, i, j) = fL_flux
                else if (SR <= 0.0) then
                    G_ext(:, i, j) = fR_flux
                else
                    SM = (PR - PL + rhoL*vL*(SL - vL) - rhoR*vR*(SR - vR)) / (rhoL*(SL - vL) - rhoR*(SR - vR))
                    if (SM >= 0.0) then
                        PM = rhoL*(vL - SL)*(vL - SM) + PL
                        U_star_L = [rhoL*(SL - vL)/(SL - SM), rhoL*(SL - vL)/(SL - SM)*uL, rhoL*(SL - vL)/(SL - SM)*SM, rhoL*(SL - vL)/(SL - SM)*(EL/rhoL + (SM - vL)*(SM + PL/(rhoL*(SL - vL))))]
                        G_ext(:, i, j) = fL_flux + SL * (U_star_L - [rhoL, rhoL*uL, rhoL*vL, EL])
                    else
                        PM = rhoR*(vR - SR)*(vR - SM) + PR
                        U_star_R = [rhoR*(SR - vR)/(SR - SM), rhoR*(SR - vR)/(SR - SM)*uR, rhoR*(SR - vR)/(SR - SM)*SM, rhoR*(SR - vR)/(SR - SM)*(ER/rhoR + (SM - vR)*(SM + PR/(rhoR*(SR - vR))))]
                        G_ext(:, i, j) = fR_flux + SR * (U_star_R - [rhoR, rhoR*uR, rhoR*vR, ER])
                    end if
                end if
            end do
        end do
        
        !$acc parallel loop collapse(2) present(rhs, F_ext, G_ext)
        do j = 1, ny
            do i = 1, nx
                rhs(:, i, j) = - (F_ext(:, i, j) - F_ext(:, i-1, j)) / dx &
                               - (G_ext(:, i, j) - G_ext(:, i, j-1)) / dy
            end do
        end do
    end subroutine compute_rhs

    subroutine write_vtk_binary(frame, state, nx, ny, ng, dx, dy)
        real, dimension(4, 1-ng:nx+ng, 1-ng:ny+ng), intent(in) :: state
        integer, intent(in) :: frame, nx, ny, ng
        real, intent(in) :: dx, dy
        character(len=32) :: filename
        integer :: i, j
        real(kind=4) :: out_val, out_u, out_v, out_w
        real :: v_right, v_left, u_top, u_bot, vort
        
        write(filename, '("khi_", i0.4, ".vtk")') frame
        ! Following Visualization Toolkit rules
        open(unit=10, file=trim(filename), status='replace', form='formatted')
        write(10, '(a)') '# vtk DataFile Version 3.0'
        write(10, '(a)') 'KHI Simulation Data'
        write(10, '(a)') 'BINARY'
        write(10, '(a)') 'DATASET STRUCTURED_POINTS'
        write(10, '(a,i0,a,i0,a)') 'DIMENSIONS ', nx, ' ', ny, ' 1'
        write(10, '(a)') 'ORIGIN 0 0 0'
        write(10, '(a)') 'SPACING 1 1 1'
        write(10, '(a,i0)') 'POINT_DATA ', nx * ny

        ! Enter all the data in binary (weightless for high resolution)
        write(10, '(a)') 'SCALARS Density float 1'
        write(10, '(a)') 'LOOKUP_TABLE default'
        close(10)
        open(unit=10, file=trim(filename), status='old', access='stream', &
             form='unformatted', position='append', convert='big_endian')
        do j = 1, ny
            do i = 1, nx
                out_val = real(state(1, i, j), 4)
                write(10) out_val
            end do
        end do
        close(10)
        open(unit=10, file=trim(filename), status='old', form='formatted', position='append')
        write(10, '(a)') 'SCALARS Z_Vorticity float 1'
        write(10, '(a)') 'LOOKUP_TABLE default'
        close(10)
        open(unit=10, file=trim(filename), status='old', access='stream', &
             form='unformatted', position='append', convert='big_endian')
        do j = 1, ny
            do i = 1, nx
                v_right = state(3, i+1, j) / state(1, i+1, j)
                v_left  = state(3, i-1, j) / state(1, i-1, j)
                u_top   = state(2, i, j+1) / state(1, i, j+1)
                u_bot   = state(2, i, j-1) / state(1, i, j-1)
                vort = (v_right - v_left) / (2.0 * dx) - (u_top - u_bot) / (2.0 * dy)
                out_val = real(vort, 4)
                write(10) out_val
            end do
        end do
        close(10)  
        open(unit=10, file=trim(filename), status='old', form='formatted', position='append')
        write(10, '(a)') 'VECTORS Velocity float'
        close(10)
        open(unit=10, file=trim(filename), status='old', access='stream', &
             form='unformatted', position='append', convert='big_endian')
        do j = 1, ny
            do i = 1, nx
                out_u = real(state(2, i, j) / state(1, i, j), 4)
                out_v = real(state(3, i, j) / state(1, i, j), 4)
                out_w = 0.0
                write(10) out_u
                write(10) out_v
                write(10) out_w
            end do
        end do
        close(10)
    end subroutine write_vtk_binary

end program KHI_GPU_Benchmark