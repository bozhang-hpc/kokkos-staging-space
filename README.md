## Kokkos Staging Spaces

Kokkos Staging Spaces (KSS) adds inter-application data exchange support to Kokkos parallel programming model. It uses Dataspaces (https://github.com/philip-davis/dataspaces) as the backend. KSS leverages in-memory stage in&out and unburdens the programmer from designing ad hoc data redistribution between applications in the complex scientific workflow.

*Note: Kokkos Staging Spaces is in an experimental development stage.*

## Dependencies

* MPI
* Kokkos (https://github.com/kokkos/kokkos)
* Dataspaces (https://github.com/philip-davis/dataspaces)

## Build
````bash
> cmake ${KSS_SOURCE_DIR} \
    -DKokkos_ROOT=${KOKKOS_INSTALL_PREFIX} \
    -DDATASPACES_ROOT=${DATASPACES_INSTALL_PREFIX} \
    -DCMAKE_CXX_COMPILER=${CXX}
````

## API

````C++
// support up to 8D
Kokkos::View<Data_t*, Kokkos::StagingSpace> v_S1("ViewStaging1D", arg_N1);
Kokkos::View<Data_t**, Kokkos::StagingSpace> v_S2("ViewStaging2D", arg_N1, arg_N2);

Kokkos::View<Data_t*, Kokkos::HostSpace> v_H1("ViewHost1D", arg_N1);

// from host to staging
Kokkos::deep_copy(v_S1, v_H1);

// from staging to host
Kokkos::deep_copy(v_H1, v_S1);
````

